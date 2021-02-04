# Remove after Ray 1.2 release.
import asyncio
from typing import Optional, Dict, Callable

import ray
from ray.util.queue import Queue as RayQueue, Empty, Full

# Remove after Ray 1.2 release.
if getattr(RayQueue, "shutdown", None) is not None:
    from ray.util.queue import _QueueActor
else:
    # Have to copy the class here so that we can subclass this for mocking.
    # If we have the @ray.remote decorator, then we can't subclass it.
    class _QueueActor:
        def __init__(self, maxsize):
            self.maxsize = maxsize
            self.queue = asyncio.Queue(self.maxsize)

        def qsize(self):
            return self.queue.qsize()

        def empty(self):
            return self.queue.empty()

        def full(self):
            return self.queue.full()

        async def put(self, item, timeout=None):
            try:
                await asyncio.wait_for(self.queue.put(item), timeout)
            except asyncio.TimeoutError:
                raise Full

        async def get(self, timeout=None):
            try:
                return await asyncio.wait_for(self.queue.get(), timeout)
            except asyncio.TimeoutError:
                raise Empty

        def put_nowait(self, item):
            self.queue.put_nowait(item)

        def put_nowait_batch(self, items):
            # If maxsize is 0, queue is unbounded, so no need to check size.
            if self.maxsize > 0 and len(items) + self.qsize() > self.maxsize:
                raise Full(f"Cannot add {len(items)} items to queue of size "
                           f"{self.qsize()} and maxsize {self.maxsize}.")
            for item in items:
                self.queue.put_nowait(item)

        def get_nowait(self):
            return self.queue.get_nowait()

        def get_nowait_batch(self, num_items):
            if num_items > self.qsize():
                raise Empty(f"Cannot get {num_items} items from queue of size "
                            f"{self.qsize()}.")
            return [self.queue.get_nowait() for _ in range(num_items)]


class Queue(RayQueue):
    def __init__(self, maxsize: int = 0,
                 actor_options: Optional[Dict] = None) -> None:
        actor_options = actor_options or {}
        self.maxsize = maxsize
        self.actor = ray.remote(_QueueActor).options(**actor_options).remote(
            self.maxsize)

    def shutdown(self):
        if getattr(RayQueue, "shutdown", None) is not None:
            super(Queue, self).shutdown()
        else:
            if self.actor:
                ray.kill(self.actor)
            self.actor = None


def _handle_queue(queue):
    # Process results from Queue.
    while not queue.empty():
        (actor_rank, item) = queue.get()
        if isinstance(item, Callable):
            item()


def process_results(training_result_futures, queue):
    not_ready = training_result_futures
    while not_ready:
        if queue:
            _handle_queue(queue)
        ready, not_ready = ray.wait(not_ready, timeout=0)
        ray.get(ready)
    ray.get(ready)

    if queue:
        # Process any remaining items in queue.
        _handle_queue(queue)
    return ray.get(training_result_futures)
