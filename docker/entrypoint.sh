#!/bin/bash
# Re-register extension so PyTorch sees it at the correct editable path.
# This also warms up HIP extension build context before vllm workers spawn.
pip install -e /opt/gfx906_fa --no-deps --quiet 2>/dev/null
exec "$@"
