

dataset=browsecomp
embedding_path=browsecomp_embeddings/${dataset}/reasonir

mkdir -p ${embedding_path}
CUDA_VISIBLE_DEVICES=2 python encode.py  \
  --output_dir=temp \
  --model_name_or_path reasonir/ReasonIR-8B \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --dataset_name Tevatron/grounded-browsecomp-decrypted \
  --encode_is_query \
  --query_prefix "<|user|>\nGiven a question, retrieve relevant passages that help answer the question\n<|embed|>\n" \
  --encode_output_path ${embedding_path}/queries.pkl


for s in $(seq -f "%02g" 0 3)
do
CUDA_VISIBLE_DEVICES=${s} python encode.py  \
  --output_dir=temp \
  --model_name_or_path reasonir/ReasonIR-8B \
  --bf16 \
  --per_device_eval_batch_size 16 \
  --dataset_name Tevatron/grounded-browsecomp-decrypted-corpus \
  --passage_prefix "<|embed|>\n" \
  --dataset_number_of_shards 4 \
  --dataset_shard_index ${s} \
  --encode_output_path ${embedding_path}/corpus-${s}.pkl &
done

### Search
```bash
mkdir -p browsecomp_results/${dataset}/reasonir
python -m tevatron.retriever.driver.search \
    --query_reps ${embedding_path}/queries.pkl \
    --passage_reps ${embedding_path}/'corpus*.pkl' \
    --depth 1000 \
    --batch_size 64 \
    --save_text \
    --save_ranking_to browsecomp_results/${dataset}/reasonir/rank.txt

# Convert to TREC format
python -m tevatron.utils.format.convert_result_to_trec --input browsecomp_results/${dataset}/reasonir/rank.txt \
                                                       --output browsecomp_results/${dataset}/reasonir/rank.trec
```
### Eval
```bash
python -m pyserini.eval.trec_eval -c -m recall -m ndcg_cut.10 /scratch3/zhu042/grounded-browsecomp/topics-qrels/qrel_positives.txt browsecomp_results/${dataset}/reasonir/rank.trec

python -m pyserini.eval.trec_eval -c -m recall -m ndcg_cut.10 /scratch3/zhu042/grounded-browsecomp/topics-qrels/qrel_golds.txt browsecomp_results/${dataset}/reasonir/rank.trec

```