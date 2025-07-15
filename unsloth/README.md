## Learning rate:
Typical Range: 2e-4 (0.0002) to 5e-6 (0.000005). 

- 🟩 For normal LoRA/QLoRA Fine-tuning, we recommend 2e-4 as a starting point. 
- 🟦 For Reinforcement Learning (DPO, GRPO etc.), we recommend 5e-6 . 
- ⬜ For Full Fine-tuning, lower learning rates are generally more appropriate.

## Epochs
Recommended: 1-3 epochs. 

For most instruction-based datasets, training for more than 3 epochs offers diminishing returns and increases the risk of overfitting.

## LoRA or QLoRA: Hyperparameters & Recommendations

| Siêu tham số | Chức năng | Cài đặt khuyến nghị |
| :--- | :--- | :--- |
| **LoRA Rank (`r`)** | Kiểm soát số lượng tham số có thể huấn luyện trong các ma trận adapter `LoRA`. `Rank` cao hơn làm tăng dung lượng mô hình nhưng cũng tăng mức sử dụng bộ nhớ. | 8, 16, 32, 64, 128\<br\>**Chọn 16 hoặc 32** |
| **LoRA Alpha (`lora_alpha`)** | Điều chỉnh độ mạnh của các điều chỉnh đã được tinh chỉnh so với `rank` (`r`). | `r` (tiêu chuẩn) hoặc `r * 2` (heuristic phổ biến). [Chi tiết hơn tại đây](https://www.google.com/search?q=%23lora-alpha-and-rank-relationship). |
| **LoRA Dropout** | Một kỹ thuật điều chuẩn (regularization) ngẫu nhiên đặt một phần các kích hoạt `LoRA` về 0 trong quá trình huấn luyện để ngăn `overfitting`. Không hữu ích lắm, vì vậy chúng tôi mặc định đặt nó là 0. | 0 (mặc định) đến 0.1 |
| **Weight Decay** | Một thuật ngữ điều chuẩn phạt các trọng số lớn để ngăn `overfitting` và cải thiện khả năng tổng quát hóa. Đừng sử dụng số quá lớn\! | 0.01 (khuyến nghị) - 0.1 |
| **Warmup Steps** | Tăng dần `learning rate` khi bắt đầu huấn luyện. | 5-10% tổng số bước |
| **Scheduler Type** | Điều chỉnh `learning rate` một cách linh hoạt trong quá trình huấn luyện. | `linear` hoặc `cosine` |
| **Seed (`random_state`)** | Một số cố định để đảm bảo khả năng tái tạo kết quả. | Bất kỳ số nguyên nào (ví dụ: `42`, `3407`) |
| **Target Modules** | Chỉ định các phần của mô hình bạn muốn áp dụng adapter `LoRA` — `attention`, `MLP`, hoặc cả hai. | `Attention: q_proj, k_proj, v_proj, o_proj`\<br\>`MLP: gate_proj, up_proj, down_proj`\<br\>**Khuyến nghị nhắm mục tiêu tất cả các lớp tuyến tính chính**: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. |


## Effective Batch Size

$$Effective\_Batch\_Size = batch\_size * gradient\_accumulation\_steps$$

- Một `Effective Batch Size` lớn hơn thường dẫn đến việc huấn luyện mượt mà và ổn định hơn. 
- Một `Effective Batch Size` nhỏ hơn có thể tạo ra nhiều phương sai hơn.

Mặc dù mỗi tác vụ đều khác nhau, cấu hình sau đây cung cấp một điểm khởi đầu tuyệt vời để đạt được `Effective Batch Size` ổn định là `16`

| Tham số | Mô tả | Cài đặt khuyến nghị |
| :--- | :--- | :--- |
| **Batch Size (`batch_size`)** | Số lượng mẫu được xử lý trong một lần truyền xuôi/ngược (forward/backward pass) trên một GPU. **Yếu tố chính ảnh hưởng đến việc sử dụng VRAM**. Giá trị cao hơn có thể cải thiện việc sử dụng phần cứng và tăng tốc độ huấn luyện, nhưng chỉ khi chúng vừa với bộ nhớ. | **2** |
| **Gradient Accumulation (`gradient_accumulation_steps`)** | Số lượng micro-batch được xử lý trước khi thực hiện một lần cập nhật trọng số mô hình. **Yếu tố chính ảnh hưởng đến thời gian huấn luyện**. Cho phép mô phỏng `batch_size` lớn hơn để tiết kiệm `VRAM`. Giá trị cao hơn làm tăng thời gian huấn luyện cho mỗi `epoch`. | **8** |
| **Effective Batch Size (Tính toán)** | `batch size` thực sự được sử dụng cho mỗi lần cập nhật gradient. Nó ảnh hưởng trực tiếp đến sự ổn định, chất lượng và hiệu suất cuối cùng của mô hình. | 4 đến 16, **Khuyến nghị: 16 (từ 2 \* 8)** |