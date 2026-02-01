旨在構建 **"FluxTrain"**—— 一個專注於消費級顯卡、零配置、自動卸載（Auto-Offload）的深度學習訓練框架。

---

# 專案計畫書：FluxTrain 自動化大模型異構訓練框架

## 1. 專案願景 (Vision)
打造一個比 DeepSpeed 更易用、比 PyTorch FSDP 更智能的訓練框架。
**核心目標**：讓單張 RTX 3090/4090 能跑起 70B 模型，讓多卡訓練無需編寫複雜 Config。

## 2. 開發週期總覽 (Timeline)
*   **總時長**：預計 24 週 (6 個月)
*   **里程碑**：
    *   M1: 單卡原型 (能跑通)
    *   M2: 性能優化 (能跑快)
    *   M3: 多卡擴展 (自動化)
    *   M4: 發布與生態 (UX/UI)

---

## 3. 詳細執行階段 (Phases)

### Phase 0: 架構設計與技術驗證
**目標**：確定底層技術路線，驗證 Python 與 CUDA 之間的記憶體搬運極限。

*   **W1: 競品分析與核心設計**
    *   深入解讀 `DeepSpeed ZeRO-3` 與 `Accelerate` 源碼，找出其配置複雜的痛點。
    *   設計 `VirtualMemoryManager` 架構：定義 Tensor 在 GPU/CPU/NVMe 間的狀態機。
*   **W2: 最小可行性驗證 (PoC)**
    *   撰寫腳本測試 `torch.cuda.Stream` 與 `Pinned Memory` 的異步傳輸頻寬。
    *   驗證 PyTorch Hooks (`register_full_backward_hook`) 攔截梯度與權重的可行性。

### Phase 1: 單卡極限生存引擎 (Weeks 3-8)
**目標**：在單張 24GB 顯卡上成功訓練 Llama-2-70B (Batch Size=1)，不求快，但求不 OOM。

*   **W3-W4: 權重與梯度卸載 (Static Offloading)**
    *   實作 `Layer-wise Loading`：僅在 Forward/Backward 計算當前層時，將權重載入 GPU。
    *   實作 `Optimizer Offload`：強制將 AdamW 狀態存於 CPU RAM。
*   **W5-W6: 動態顯存管理 (Dynamic Eviction)**
    *   開發 `MemoryMonitor`：實時監控 GPU 顯存水位。
    *   實作「LRU 驅逐策略」：當顯存 >90% 時，自動將最久未用的 Activation 搬移至 CPU。
*   **W7-W8: 封裝與 API 設計**
    *   開發 `fluxtrain.auto_wrap(model)` 接口。
    *   完成 HuggingFace Transformers 的無縫整合測試。

### Phase 2: 速度優化與硬體感知
**目標**：解決 Phase 1 的「慢」問題，利用流水線技術隱藏 IO 延遲。

*   **W9-W10: 預取機制 (Prefetching)**
    *   分析計算圖，預測下一層所需的權重。
    *   實作 `Async Prefetcher`：在計算 Layer N 時，後台 Stream 預先加載 Layer N+1。
*   **W11-W12: 硬體感知模組 (Hardware Awareness)**
    *   開發 `Profiler`：啟動時自動測試 PCIe 頻寬與 NVMe 讀寫速度。
    *   實作「自適應策略」：根據頻寬自動決定 Offload 的激進程度 (Aggressiveness)。
*   **W13-W14: 量化整合 (Quantization)**
    *   原生整合 8-bit / 4-bit Optimizer (基於 `bitsandbytes`)。
    *   實作混合精度 (BF16/FP16) 的自動轉換邏輯。

### Phase 3: 多卡擴展與自動分片
**目標**：超越 DeepSpeed 的易用性，實現「插卡即用」的分散式訓練。

*   **W15-W16: 自動分片 (Auto-Sharding)**
    *   基於 PyTorch FSDP 封裝，但移除手動 `wrapping_policy`。
    *   算法自動遍歷 Module Tree，識別 Transformer Block 並進行最優切分。
*   **W17-W18: 通訊與計算重疊 (Comm-Compute Overlap)**
    *   優化多卡間的 `AllGather` 與 `ReduceScatter`，使其與 CPU Offload 的 IO 操作並行。
*   **W19-W20: 異構集群支援**
    *   支援非對稱顯卡（例如一張 3090 + 一張 4090）的負載平衡（Load Balancing），根據顯存大小分配不同數量的 Layers。

### Phase 4: 使用者體驗與發布
**目標**：打造「開發者友善」的工具，準備開源。

*   **W21: TUI 儀表板 (Dashboard)**
    *   使用 `Rich` 或 `Textual` 庫開發終端機監控介面。
    *   顯示：即時 VRAM 使用率、PCIe 流量、預估剩餘時間、當前 Offload 狀態。
*   **W22: 容錯與優雅降級 (Graceful Degradation)**
    *   攔截 CUDA OOM 錯誤，自動觸發「緊急卸載到 Disk」而非崩潰。
    *   提供建議日誌：「檢測到頻繁 Swap，建議減少 Batch Size 至 X」。
*   **W23: 文檔與範例**
    *   撰寫 `README.md`，提供 "Zero to Hero" 的 Colab 範例。
    *   製作對比 DeepSpeed 的 Benchmark 報告。
*   **W24: v0.1 Alpha 發布**
    *   發布至 PyPI。
    *   在 Reddit (r/LocalLLaMA) 與 Twitter 發布宣傳。

---

## 4. 資源需求 (Resources)

### 硬體設備
*   **開發機**：
    *   CPU: 高核心數 (如 Threadripper 或 EPYC) 以測試 CPU Offload 瓶頸。
    *   RAM: 128GB+ DDR4/DDR5 (模擬大模型權重存放)。
    *   GPU: 至少 2 張不同型號顯卡 (例如 1x RTX 3090, 1x RTX 4090) 以測試異構兼容性。
    *   Storage: 高速 NVMe SSD (Gen4/Gen5) 測試 Disk Offload。

### 技術棧
*   **核心語言**: Python (PyTorch), C++ (CUDA Extensions for custom kernels if needed).
*   **依賴庫**: `torch.distributed`, `bitsandbytes`, `safetensors`, `accelerate`.

---

## 5. 風險評估 (Risk Management)

| 風險項目 | 可能性 | 影響程度 | 應對策略 |
| :--- | :--- | :--- | :--- |
| **PCIe 頻寬成為絕對瓶頸** | 高 | 高 | 引入更激進的 Activation Checkpointing (重算換 IO)；支援 NVLink (若有)。 |
| **PyTorch 版本更新導致 Hook 失效** | 中 | 中 | 建立 CI/CD 流程，每週針對 PyTorch Nightly 進行測試；盡量使用穩定 API。 |
| **推理/訓練速度過慢** | 中 | 高 | 提供「預估時間」功能，讓用戶有心理準備；強調「能跑」大於「跑快」的定位。 |

---

## 6. 成功指標 (KPIs)
1.  **Zero Config**: 用戶只需 `import fluxtrain` 和 `model = fluxtrain.wrap(model)` 兩行代碼。
2.  **Hardware Efficiency**: 在 RTX 3090 (24GB) 上能 Fine-tune Llama-3-70B (QLoRA)。
3.  **Scalability**: 多卡訓練效率達到線性擴展的 80% 以上。
