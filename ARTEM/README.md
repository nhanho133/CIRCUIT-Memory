# ARTEM pipeline (EPBench, LLM-extracted cue + calibrated confidence)

Code cho chuỗi thực nghiệm EPBench của CIRCUIT Memory: bridge dữ liệu EPBench, LLM tự trích
events/cue+confidence, isotonic calibration per-axis (held-out), build candidate set theo
4 method (hard/gated/literal/circuit), answer+judge qua LLM, tổng hợp báo cáo.

## Chạy (cần data EPBench + Modal endpoint riêng, xem code để biết contract JSON)

```bash
python bridge_epbench_to_artem.py --universe <universe> --chapters <n> --out data_root_ep200 --book_id 1
export MODAL_LLM_URL="https://<your-endpoint>.modal.run"
python llm_event_extraction.py --url $MODAL_LLM_URL
python cue_extractor.py --base data_root_ep200 --book_id 1 --backend modal
python calibrate_conf.py --base data_root_ep200 --book_id 1
python build_circuit_llmevents_results.py
python run_artem_llm.py --base data_root_extracted_circuit_llmev668 --book_id 1 --backend modal
python report_all_b4.py
```

## Demo nhanh (không cần data/GPU)

```bash
python gated_novelty_demo.py
```
Minh họa cơ chế novelty (confidence-gated intersection) trên vài event mẫu tự chứa, chạy
được ngay không cần EPBench/Modal.

## requirements

```
numpy pandas scikit-learn sentence-transformers modal torch transformers
```
