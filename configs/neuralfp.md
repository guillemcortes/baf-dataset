NeuralFP configuration files can be found in its repository: i.e. https://github.com/guillemcortes/neural-audio-fp/blob/main/config/custom_baf_paper.yaml

The configuration file looks like this excerpt:
```yml
# Index Parameters
INDEX:
    INDEX_TYPE: 'ivfpq'
    N_CENTROIDS: 256
    CODE_SZ: 64
    NBITS: 8
    NPROBE: 40

# Matcher Parameters
MATCHER:
    INDEX_NN: 5  # Number of nearest neighbors the index return when querying.
    CHUNK_LENGTH: 9  # Number of segments in a query chunk. (9 segments ==  5 seconds)
    OVERLAP: 7  # number of segments to overlap
    THRESHOLD: 0.65  # score threshold
```