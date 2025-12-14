from time import perf_counter, sleep


def wait_for_next_tick(tick: float, dt: float) -> float:
    next_tick = tick + dt
    sleep_time = next_tick - perf_counter()

    if sleep_time > 0:
        sleep(sleep_time)

    return perf_counter()
