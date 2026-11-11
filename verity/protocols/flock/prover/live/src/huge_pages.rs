//! [`HugePages`]: the system allocator, asking for transparent huge pages on every allocation of [`MIN_BYTES`] or more before
//! its first touch. The prover's tables are fresh vectors of 2^k field elements each rep (the lincheck's eq table and folds,
//! the encoded rows), each mapped anew by glibc and faulted in by the kernel 4 KiB at a time, the prover's threads waiting on
//! each other's faults; huge pages fault each 2 MiB once. The kernel takes the advice where `transparent_hugepage/enabled`
//! is `madvise` or `always`, and ignores it under `never`. A GPU prover turns the advice off ([`set`]): its host staging
//! of each witness runs slower on advised pages than on small ones, which outweighs what the advice saves it. Without the
//! advice each such allocation is bound to local allocation instead (`mbind`, `MPOL_LOCAL`): its pages land where they would
//! by default, on the faulting thread's node, but a policy set on a mapping takes it out of NUMA balancing, whose hint faults
//! on so many fresh small pages at times stall every thread of a walk for many times its length.
use std::alloc::{GlobalAlloc, Layout, System};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};

/// glibc's largest mmap threshold: an allocation this large is a mapping of its own, which the advice or the policy covers
/// whole (a part of a mapping would split it, and glibc's `realloc` could no longer `mremap` it).
pub const MIN_BYTES: usize = 32 << 20;
const PAGE: usize = 4096;
const MADV_HUGEPAGE: i32 = 14;
const MPOL_LOCAL: libc::c_int = 4;

unsafe extern "C" {
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

pub struct HugePages;

static ON: AtomicBool = AtomicBool::new(true);

/// Whether later allocations take the advice; set before the prover's first large allocation.
pub fn set(on: bool) {
    ON.store(on, Ordering::Relaxed);
}

fn advise(p: *mut u8, size: usize) -> *mut u8 {
    if !p.is_null() && size >= MIN_BYTES {
        let (a, e) = (p as usize / PAGE * PAGE, (p as usize + size).next_multiple_of(PAGE));
        // advice: refused, the pages stay small; the policy: refused (a kernel without NUMA), they stay NUMA balancing's
        if ON.load(Ordering::Relaxed) {
            unsafe { madvise(a as *mut c_void, e - a, MADV_HUGEPAGE) };
        } else {
            let none = std::ptr::null::<libc::c_ulong>();
            unsafe { libc::syscall(libc::SYS_mbind, a, e - a, MPOL_LOCAL, none, 0 as libc::c_ulong, 0 as libc::c_uint) };
        }
    }
    p
}

unsafe impl GlobalAlloc for HugePages {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        advise(unsafe { System.alloc(l) }, l.size())
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        advise(unsafe { System.alloc_zeroed(l) }, l.size())
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        advise(unsafe { System.realloc(p, l, n) }, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `/proc/self/smaps` mapping holding `p`: its range and `VmFlags`.
    fn mapping(p: usize) -> (usize, usize, String) {
        let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
        let mut cur = None;
        for line in smaps.lines() {
            if let Some((range, _)) = line.split_once(' ').filter(|(r, _)| r.contains('-') && !r.ends_with(':')) {
                let (a, e) = range.split_once('-').unwrap();
                cur = Some((usize::from_str_radix(a, 16).unwrap(), usize::from_str_radix(e, 16).unwrap()));
            } else if let (Some(flags), Some((a, e))) = (line.strip_prefix("VmFlags:"), cur) {
                if a <= p && p < e {
                    return (a, e, flags.to_string());
                }
            }
        }
        panic!("no mapping holds {p:#x}");
    }

    /// The memory policy `/proc/self/numa_maps` gives the mapping starting at `a`; None on a kernel without NUMA.
    fn policy(a: usize) -> Option<String> {
        let maps = std::fs::read_to_string("/proc/self/numa_maps").ok()?;
        let line = maps.lines().find(|l| l.split_whitespace().next() == Some(&format!("{a:x}")))?;
        line.split_whitespace().nth(1).map(str::to_string)
    }

    #[test]
    fn a_large_allocation_is_one_mapping_through_realloc_advised_or_else_bound_to_local_allocation() {
        let l = Layout::from_size_align(MIN_BYTES + 12345, 16).unwrap();
        unsafe {
            let p = HugePages.alloc_zeroed(l);
            let (a, e, flags) = mapping(p as usize);
            assert!(flags.split_whitespace().any(|f| f == "hg"), "not advised: {flags}");
            assert!(a <= p as usize && p as usize + l.size() <= e, "the allocation spans mappings");
            assert!(policy(a).is_none_or(|m| m == "default"), "an advised mapping bound: {:?}", policy(a));
            *p.add(l.size() - 1) = 7;
            let n = 2 * MIN_BYTES;
            let q = HugePages.realloc(p, l, n);
            let (a, e, flags) = mapping(q as usize);
            assert!(flags.split_whitespace().any(|f| f == "hg") && a <= q as usize && q as usize + n <= e);
            assert_eq!(*q.add(l.size() - 1), 7);
            HugePages.dealloc(q, Layout::from_size_align(n, 16).unwrap());
            let small = Layout::from_size_align(1 << 20, 16).unwrap();
            let s = HugePages.alloc(small);
            assert!(!s.is_null());
            HugePages.dealloc(s, small);
            set(false);
            let p = HugePages.alloc_zeroed(l);
            let (a, e, flags) = mapping(p as usize);
            assert!(!flags.split_whitespace().any(|f| f == "hg"), "advised while off: {flags}");
            assert!(a <= p as usize && p as usize + l.size() <= e, "the allocation spans mappings");
            assert!(policy(a).is_none_or(|m| m == "local"), "not bound to local allocation: {:?}", policy(a));
            *p.add(l.size() - 1) = 7;
            let q = HugePages.realloc(p, l, n);
            set(true);
            let (a, e, _) = mapping(q as usize);
            assert!(a <= q as usize && q as usize + n <= e && policy(a).is_none_or(|m| m == "local"), "{:?}", policy(a));
            assert_eq!(*q.add(l.size() - 1), 7);
            HugePages.dealloc(q, Layout::from_size_align(n, 16).unwrap());
        }
    }
}
