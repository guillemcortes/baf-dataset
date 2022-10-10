
# PeakFP

Audio Fingerprinting Baseline consisting in comparing spectrogram from queries and reference and check the peaks in common.


## Documentation

The PeakFP pipeline consists in running 3 main scripts:

**extractor.py**  &rarr; Extracts the fingerprints  
``` bash
usage: extractor.py [--spectrogram_path SPECTROGRAM_PATH]
                    audio_path peakfp_path

positional arguments:
  audio_path            Origin audio path (any format accepted by librosa).
  peakfp_path           Destination fingerprint path.

optional arguments:
  --spectrogram_path SPECTROGRAM_PATH
                        Optional destination spectrogram path (debug purposes).
```

**indexer.py**  &rarr; Creates the index  
``` bash
usage: indexer.py peakfps_list_path index_path

positional arguments:
  peakfps_list_path  Path of separated list of fingerprints paths.
  index_path         Destination index path.
```

**matcher.py**  &rarr; Matches (1 vs Index) a fingerprint query against de index.  
``` bash
usage: matcher.py query_fp_path index_path

positional arguments:
  query_fp_path  Path of query peakfp fingerprint.
  index_path     Index path.
```
## Roadmap

- Create tests

- Make it configurable by config file
