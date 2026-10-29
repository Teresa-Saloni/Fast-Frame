# bench/data/

Fixed benchmark workload (git-ignored, ~30 MB):

    traffic_1080p_30s.mp4   900 frames, 1920x1080, H.264, 30 fps

Any 30-second 1080p clip works; record which one you used in BENCHMARKS.md and never change it
between rows. CI uses the synthetic source instead (`--synthetic 600 --size 1920x1080 --fake-cost 15000`),
which exercises every stage except the real decoder and model.
