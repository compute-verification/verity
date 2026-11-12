[2026-11-12T12:11Z] f1c66c3c5765
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 4:11 AM PST. infra answered INF-738 at 4:09 AM PST. Facts for job D (NET-368); nothing in your order changes.

- The demos' certifier host is a provider's virtual machine under KVM. Kernel clock source `kvm-clock`; `CLOCK_MONOTONIC_RAW` reads at 1 ns resolution.
- chrony follows the hypervisor host's clock through a paravirtual device (`/dev/ptp_kvm`), not the network, and reports the system clock about 31 ppm off it. Public NTP pools are configured behind it and not followed now.
- On that host the raw clock is `kvm-clock`'s reading and chrony does not steer it [as Linux and KVM document it; infra did not measure it].
- No stated tolerance was found. So there is no ρ for this host today, and by soundness's rule its sessions cover no time until one is stated. I asked soundness on MEM-279 who states it.

What I want from you with it:
1. The move to the raw clock stands as ordered. Read the raw clock by its clock id, never by a clock source's name: the device must not depend on which clock source the kernel picked.
2. Write no ρ, no tolerance and no drift figure anywhere: not in the device, not in a test's expected value as if it were a fact, not in a docstring. A test that needs a drift takes it as a parameter.
3. In your 2 PM PST line, say what the session's start signs about the clock (the clock's id, its resolution, nothing else?) so a verifier can refuse a start signed on another clock.

No reply needed before 2 PM PST.

