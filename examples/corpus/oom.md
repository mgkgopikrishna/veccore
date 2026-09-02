A container killed for exceeding its memory limit exits with code 137. The kernel OOM
killer terminates it immediately with no graceful shutdown, so in-flight requests are
dropped. Raising the limit unblocks the workload, but if the process grows without
bound it is a leak and a higher limit only delays the kill.
