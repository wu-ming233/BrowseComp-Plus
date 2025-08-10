mkdir -p retrieval/processing

python -m tevatron.retriever.driver.encode \
  --model_name_or_path Qwen/Qwen3-Embedding-8B \
  --dataset_name Tevatron/grounded-browsecomp-decrypted \
  --encode_output_path retrieval/processing/qwen3-8b_query.pkl \
  --query_max_len 512 \
  --encode_is_query \
  --normalize \
  --pooling eos \
  --query_prefix "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:" \
  --per_device_eval_batch_size 156 \
  --fp16

python -m tevatron.retriever.driver.search --query_reps retrieval/processing/qwen3-8b_query.pkl --passage_reps 'indexes/qwen3-embedding-8b/corpus.shard*.pkl' --depth 1000 --batch_size 128 --save_text --save_ranking_to retrieval/processing/qwen3-8b_top1000.txt

python scripts_run_retrieval/convert_tevatron_to_trec.py --input retrieval/processing/qwen3-8b_top1000.txt --output retrieval/qwen3-8b_top1000.trec --tag qwen3-8b

echo "Qwen3-8B Retrieval Results (Positive):"
python -m pyserini.eval.trec_eval  -c -m recall.100,1000  -m ndcg_cut.10   topics-qrels/qrel_positives.txt  retrieval/qwen3-8b_top1000.trec

echo "Qwen3-8B Retrieval Results (Gold):"
python -m pyserini.eval.trec_eval  -c -m recall.100,1000  -m ndcg_cut.10   topics-qrels/qrel_golds.txt  retrieval/qwen3-8b_top1000.trec