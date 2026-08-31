# InvestRAG experiment exp_f1ea76a505d3

- Status: **completed**
- Dataset: `acceptance-demo-2026-08-30`
- Track: `portable`
- Corpus chunks: 14

## Metrics

- `faiss/hybrid/success@10`: 1.0 score
- `faiss/hybrid/recall@10`: 1.0 score
- `faiss/hybrid/mrr@10`: 1.0 score
- `faiss/hybrid/ndcg@10`: 0.9887 score
- `faiss/hybrid/abstention-accuracy`: 1.0 score
- `faiss/hybrid/retrieval-latency-p95-ms`: 172.534 milliseconds
- `chroma/hybrid/success@10`: 1.0 score
- `chroma/hybrid/recall@10`: 1.0 score
- `chroma/hybrid/mrr@10`: 1.0 score
- `chroma/hybrid/ndcg@10`: 0.9887 score
- `chroma/hybrid/abstention-accuracy`: 1.0 score
- `chroma/hybrid/retrieval-latency-p95-ms`: 163.624 milliseconds
- `qdrant/hybrid/success@10`: 1.0 score
- `qdrant/hybrid/recall@10`: 1.0 score
- `qdrant/hybrid/mrr@10`: 1.0 score
- `qdrant/hybrid/ndcg@10`: 0.9887 score
- `qdrant/hybrid/abstention-accuracy`: 1.0 score
- `qdrant/hybrid/retrieval-latency-p95-ms`: 165.104 milliseconds

## Reproducibility

```json
{
  "schema_version": "1",
  "corpus_checksum": "1266fcb75e74a2ed4e82b8755b3591cc768273e445615101e1f4bf90b6420282",
  "source_version_ids": [
    "version_4f4e0f41d522ac1f8328",
    "version_39a524acb681b0b3112c",
    "version_367beb97401bcaa28595",
    "version_5a3699d6447536329bee",
    "version_ca608783460b5c134068",
    "version_02c1be3fbb89d5c64261",
    "version_9f3f9640ba20c0e295e2",
    "version_4e3a790e01351faebe36",
    "version_e82a2afb2c0c05609c9c",
    "version_93c1b98346b7ee3654d0",
    "version_a074f165f2acf7d1d954"
  ],
  "source_checksums": {
    "version_4f4e0f41d522ac1f8328": "3376e8bdd6d9470399053f461801fae0eea07cfadf95955382047eb2fa31c45d",
    "version_39a524acb681b0b3112c": "cfa99c39e3d567d13141cb094d06287b37a90c1bc0ae1e47f5b6be4955963bbb",
    "version_367beb97401bcaa28595": "0f4b90b44fc225a4e660b19f32c2bf6520dc6b61d7b9416e54e1526cc2da75b6",
    "version_5a3699d6447536329bee": "941f2ed64c184bc136f1b68394267c7da2ef264cac8f885e7e11e1328d932b79",
    "version_ca608783460b5c134068": "50da96053a296a26680a82be2ef0f315413de776aa55e1dbdd24bf61c6b28e75",
    "version_02c1be3fbb89d5c64261": "0e8db1fa278f33c1e7354decb91f88352f98f455410d657dac3936e998323265",
    "version_9f3f9640ba20c0e295e2": "7e8df84c8f7e2f165b2a664ef638bcb179136e94a20435350de4ac89f97df180",
    "version_4e3a790e01351faebe36": "8de8342483f907556a343e891045803915585f57a442a44d8e9b1825b5755966",
    "version_e82a2afb2c0c05609c9c": "6077cb73268d8539ec2d2915cdf6c00181c7a5798e214abbd15c2910eafd0d1c",
    "version_93c1b98346b7ee3654d0": "7a665f4fed60742ddead391c1e91484eb934bca534d9b9d7b971ea5bd0838826",
    "version_a074f165f2acf7d1d954": "1b3ad4e43659f49671c9c54706165e7d57fa0224acec847c22d7f866972991dc"
  },
  "embedding_model": "BAAI/bge-m3",
  "embedding_dimension": 1024,
  "embedding_fallback": false,
  "chunk_profiles": [
    "hybrid"
  ],
  "retrieval": {
    "top_k": 10,
    "warmup_runs": 1,
    "repetitions": 1,
    "rrf_k": 60,
    "evidence_gate": "non-stopword lexical overlap"
  },
  "vector_stores": [
    "faiss",
    "chroma",
    "qdrant"
  ],
  "seed": "not-applicable: deterministic exact/local retrieval",
  "python": "3.12.13 (main, Mar 24 2026, 22:57:53) [MSC v.1944 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.26200-SP0",
  "dependencies": {
    "numpy": "2.5.2",
    "faiss-cpu": "1.15.0",
    "chromadb": "1.5.9",
    "qdrant-client": "1.19.0",
    "langchain-core": "1.6.1"
  }
}
```
