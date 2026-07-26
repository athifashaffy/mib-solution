#!/bin/sh
set -eu
exec python3 -m mib.main "$1" "$2"
