



Usage: normal_to_height.exe [OPTIONS] [PATHS]...

Arguments:
  [PATHS]...

Options:
  -h, --height
          Should calculate height/displacement/parallax map? Will be save with _p suffix
  -a, --ao
          Should calculate ambient occlusion map? Will be save with _ap suffix
  -g, --global
          Computes downscaled passes to better approximate global differences in height
  -i, --iters <NOISY_ITERATIONS>
          The number of passes each pixel gets while computing noisy height [default: 4]
  -r, --range <NOISY_RANGE>
          The range of the ray used in noisy height [default: 21]
  -o, --overlap <OVERLAPPED_ITERATIONS>
          The number of passes when computing height over the axes. Removes tiling issues [default: 5]
      --precise
          Saves height to a 16bit PNG instead of 8bit
      --decay <DECAY>
          Exponential decay when computing height over the axes. Higher = faster decay [default: 128]
      --ao_decay <AO_DECAY>
          [default: 256]
  -b, --blur <BLUR>
          The amount (sigma value) of gaussian blur applied to height [default: 1]
      --dx

  -d, --downscale <DOWNSCALE>
          [default: 1]
  -h, --help
          Print help
  -V, --version
          Print version
