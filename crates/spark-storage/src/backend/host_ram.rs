// SPDX-License-Identifier: AGPL-3.0-only
// Host-RAM storage backend for KVFlash: cold KV chunks live in a pinned-host
// store keyed by GroupKey, paged back to device on demand. Mirrors the
// PosixBackend bounce-buffer pattern but trades NVMe for host memory.

use anyhow::{Result, bail};
use std::collections::HashMap;
use std::ffi::c_void;

use super::{ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::GroupKey;

pub struct HostRamBackend {
    group_bytes: usize,
    bounce: PinnedBuffer,
    store: HashMap<GroupKey, Vec<u8>>,
}

impl HostRamBackend {
    pub fn new(group_bytes: usize) -> Result<Self> {
        // PinnedBuffer::new needs a live CUDA context (cuMemAllocHost_v2);
        // callers must have initialized one (see tests).
        let bounce = PinnedBuffer::new(group_bytes)?;
        Ok(Self {
            group_bytes,
            bounce,
            store: HashMap::new(),
        })
    }
    pub fn group_bytes(&self) -> usize {
        self.group_bytes
    }
    /// Number of distinct groups currently materialized in the host store.
    pub fn len(&self) -> usize {
        self.store.len()
    }
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
    /// Test/diag accessor: true if `key` is materialized in the host store.
    pub fn contains(&self, key: GroupKey) -> bool {
        self.store.contains_key(&key)
    }
}

impl StorageBackend for HostRamBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        // Single shared pinned bounce buffer (like PosixBackend): each read
        // copies stored bytes -> bounce -> device, syncing per request so the
        // next request's copy into bounce doesn't race the in-flight DMA.
        let bytes = self.group_bytes;
        for req in requests {
            let Some(src) = self.store.get(&req.group) else {
                bail!(
                    "HostRamBackend::read: group {:?} not materialized",
                    req.group
                );
            };
            if src.len() != bytes {
                bail!(
                    "HostRamBackend::read: stored len {} != group_bytes {}",
                    src.len(),
                    bytes
                );
            }
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), self.bounce.ptr as *mut u8, bytes);
            }
            copy_h_to_d_async(
                req.dst_dev_ptr,
                self.bounce.ptr as *const c_void,
                bytes,
                stream,
            )?;
            stream_sync(stream)?;
        }
        Ok(())
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        if src.len() != self.group_bytes {
            bail!(
                "HostRamBackend::write_from_host: src len {} != group_bytes {}",
                src.len(),
                self.group_bytes
            );
        }
        self.store.insert(key, src.to_vec());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::KvKind;

    /// Pure-host unit test (no GPU required): validates that the host store's
    /// keying logic is sound. `HostRamBackend` itself cannot be constructed
    /// without a CUDA context (PinnedBuffer calls cuMemAllocHost_v2), so we
    /// exercise the exact `HashMap<GroupKey, Vec<u8>>` shape the backend uses
    /// directly: two distinct keys must not collide, and overwriting a key
    /// replaces its payload without touching siblings.
    #[test]
    fn store_keying_is_sound() {
        let mut store: HashMap<GroupKey, Vec<u8>> = HashMap::new();
        let k0 = GroupKey::new(0, 1, 0, KvKind::V);
        let k1 = GroupKey::new(0, 1, 0, KvKind::K);
        let k2 = GroupKey::new(3, 7, 2, KvKind::V);

        // Distinct keys don't collide.
        store.insert(k0, vec![0xAA; 16]);
        store.insert(k1, vec![0xBB; 16]);
        store.insert(k2, vec![0xCC; 16]);
        assert_eq!(store.len(), 3);

        // Same key overwrites in place, leaving siblings untouched.
        store.insert(k0, vec![0x11; 32]);
        assert_eq!(store.len(), 3);
        assert_eq!(store.get(&k0).unwrap().len(), 32);
        assert_eq!(*store.get(&k1).unwrap(), vec![0xBB; 16]);
        assert_eq!(*store.get(&k2).unwrap(), vec![0xCC; 16]);
    }

    #[test]
    #[ignore = "requires GPU"]
    fn write_then_read_round_trip() {
        // CUDA must be initialised before any pinned-host allocation.
        let ctx = crate::cuda_min::CudaCtx::new(0).expect("cuda init");
        let group_bytes = 4096_usize;
        let mut backend = HostRamBackend::new(group_bytes).unwrap();
        let pat: Vec<u8> = (0..group_bytes).map(|i| (i & 0xFF) as u8).collect();
        let key = GroupKey::new(0, 1, 0, KvKind::V);
        backend.write_from_host(key, &pat).unwrap();

        let dev = crate::cuda_min::DeviceBuffer::new(group_bytes).unwrap();
        let req = ReadRequest {
            group: key,
            dst_dev_ptr: dev.ptr,
        };
        backend.read(&[req], ctx.stream).unwrap();

        let mut host_back = vec![0_u8; group_bytes];
        crate::cuda_min::copy_d_to_h_async(
            host_back.as_mut_ptr() as *mut c_void,
            dev.ptr,
            group_bytes,
            ctx.stream,
        )
        .unwrap();
        crate::cuda_min::stream_sync(ctx.stream).unwrap();
        assert_eq!(host_back, pat);
    }
}
