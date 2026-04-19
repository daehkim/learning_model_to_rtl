"""
Step 4 — tinygrad: follow one op from Python tensor to scheduled execution.
Read the tinygrad source for Tensor.__matmul__ / lazy scheduling when this runs.
"""
from __future__ import annotations

from tinygrad import Tensor


def main() -> None:
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[1.0, 0.0], [0.0, 1.0]])
    c = a @ b
    # Materialize; tinygrad schedules lazily until numpy() or backward, etc.
    print("c =\n", c.numpy())


if __name__ == "__main__":
    main()
