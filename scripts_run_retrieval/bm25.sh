python -m pyserini.search.lucene  --index indexes/bm25/   --topics topics-qrels/queries.tsv   --output retrieval/bm25_top1000.trec   --bm25

echo "BM25 Retrieval Results (Positive):"
python -m pyserini.eval.trec_eval  -c -m recall.100,1000  -m ndcg_cut.10   topics-qrels/qrel_positives.txt  retrieval/bm25_top1000.trec

echo "BM25 Retrieval Results (Gold):"
python -m pyserini.eval.trec_eval  -c -m recall.100,1000  -m ndcg_cut.10   topics-qrels/qrel_golds.txt  retrieval/bm25_top1000.trec