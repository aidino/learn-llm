# Logging

Vì các thuật toán học tăng cường (reinforcement learning) trong lịch sử vốn đã khó gỡ lỗi (debug), việc chú ý cẩn thận đến việc ghi nhật ký là rất quan trọng.
Theo mặc định, các `trainer` của TRL như [`PPOTrainer`] và [`GRPOTrainer`] lưu rất nhiều thông tin liên quan đến các công cụ theo dõi thử nghiệm (experiment trackers) được hỗ trợ như Weights & Biases (wandb) hoặc TensorBoard.

Khi khởi tạo, hãy truyền tham số `report_to` vào đối tượng cấu hình tương ứng (ví dụ: [`PPOConfig`] cho `PPOTrainer`, hoặc [`GRPOConfig`] cho `GRPOTrainer`):

```python
# Dành cho PPOTrainer
ppo_config = PPOConfig(
    # ...,
    report_to="wandb"  # hoặc "tensorboard"
)

# Dành cho GRPOTrainer
grpc_config = GRPOConfig(
    # ...,
    report_to="wandb"  # hoặc "tensorboard"
)
```

Nếu bạn muốn ghi nhật ký với TensorBoard, bạn cũng có thể cần chỉ định thư mục lưu nhật ký, ví dụ, bằng cách thêm `logging_dir=PATH_TO_LOGS` vào đối tượng cấu hình (ví dụ: `PPOConfig` hoặc `GRPOConfig`).

## PPO Logging (Ghi nhật ký PPO)

Dưới đây là giải thích ngắn gọn cho các chỉ số (metrics) được ghi lại trong dữ liệu:

  * `eps`: Theo dõi số lượng `episodes` (tập) mỗi giây.
  * `objective/kl`: Giá trị trung bình của phân kỳ Kullback-Leibler (KL) giữa `policy` hiện tại và `policy` tham chiếu (reference policy).
  * `objective/entropy`: Entropy trung bình của `policy`, cho biết mức độ ngẫu nhiên của các hành động được `policy` lựa chọn.
  * `objective/non_score_reward`: Phần thưởng trung bình từ các nguồn không liên quan đến điểm số, về cơ bản là `beta * kl.sum(1)`, trong đó `beta` là hệ số phạt KL và `kl` là phân kỳ KL trên mỗi `token`.
  * `objective/rlhf_reward`: Phần thưởng RLHF trung bình, bằng `score - non_score_reward`.
  * `objective/scores`: Điểm số trung bình được trả về bởi mô hình phần thưởng (reward model) / môi trường (environment).
  * `policy/approxkl_avg`: Phân kỳ KL xấp xỉ trung bình giữa các `PPO policy` liên tiếp. Lưu ý rằng chỉ số này không giống với `objective/kl`.
  * `policy/clipfrac_avg`: Tỷ lệ trung bình các cập nhật `policy` bị cắt bớt (`clipped`), cho biết tần suất các cập nhật `policy` bị giới hạn để ngăn những thay đổi lớn.
  * `loss/policy_avg`: Giá trị `loss` trung bình của `policy`, cho biết `policy` đang hoạt động tốt như thế nào.
  * `loss/value_avg`: Giá trị `loss` trung bình của `value`, cho biết sự khác biệt giữa giá trị dự đoán và phần thưởng thực tế.
  * `val/clipfrac_avg`: Tỷ lệ trung bình các cập nhật `value function` bị cắt bớt, tương tự như `policy/clipfrac_avg` nhưng dành cho `value function`.
  * `policy/entropy_avg`: Entropy trung bình của `policy` trong quá trình huấn luyện, cho biết mức độ đa dạng của các hành động của `policy`.
  * `val/ratio`: Tỷ lệ trung bình giữa xác suất của `policy` hiện tại và xác suất của `policy` cũ, cung cấp một thước đo về mức độ thay đổi của `policy`.
  * `val/ratio_var`: Phương sai của `val/ratio`, cho biết sự biến thiên trong các thay đổi của `policy`.
  * `val/num_eos_tokens`: Số lượng `token` kết thúc chuỗi (end-of-sequence - EOS) được tạo ra, có thể cho biết số lượng phản hồi hoàn chỉnh.
  * `lr`: Tốc độ học (learning rate) hiện tại được `optimizer` sử dụng.
  * `episode`: Số `episode` hiện tại trong quá trình huấn luyện.

### Các giá trị quan trọng

Trong quá trình huấn luyện, nhiều giá trị được ghi lại, dưới đây là những giá trị quan trọng nhất:

1.  `objective/scores`: Điểm số trung bình được trả về bởi mô hình phần thưởng / môi trường.
2.  `objective/rlhf_reward`: Phần thưởng RLHF trung bình. Đây là mục tiêu cuối cùng của quá trình huấn luyện RLHF. Nếu việc huấn luyện diễn ra như dự định, chỉ số này sẽ tiếp tục tăng lên.
3.  `objective/non_score_reward`: Phần thưởng trung bình từ các nguồn không liên quan đến điểm số (ví dụ: hình phạt KL).

Dưới đây là một số tham số hữu ích để theo dõi sự ổn định (khi chúng phân kỳ hoặc giảm về 0, hãy thử điều chỉnh các biến):

1.  `loss/value_avg`: Giá trị `loss` trung bình của `value`. Nó sẽ tăng vọt / thành `NaN` khi quá trình huấn luyện không ổn.
2.  `val/ratio`: Tỷ lệ trung bình giữa xác suất của `policy` hiện tại và `policy` cũ. Con số này nên dao động quanh mức 1.0. Nếu `ratio` này quá cao (ví dụ: 2.0 hoặc 1000.0) hoặc quá nhỏ (ví dụ: 0.1), điều đó có nghĩa là các cập nhật giữa các `policy` liên tiếp là quá đột ngột.
3.  `policy/clipfrac_avg` và `policy/approxkl_avg`: Nếu `val/ratio` quá cao, `ratio` sẽ bị cắt bớt, dẫn đến `policy/clipfrac_avg` và `policy/approxkl_avg` cũng cao theo.
4.  `objective/kl`: Phân kỳ KL trung bình. Nó nên giữ ở mức dương và lý tưởng là không quá lớn, để `policy` không đi quá xa so với `reference policy`.

-----

## GRPO Logging (Ghi nhật ký GRPO)

Dưới đây là giải thích ngắn gọn cho các chỉ số được ghi lại trong dữ liệu cho `GRPO trainer`:

  * `num_tokens`: Tổng số `token` đầu vào đã được xử lý trong quá trình huấn luyện cho đến nay.

**Completions (Các chuỗi hoàn thành):**

  * `completions/mean_length`: Độ dài trung bình của tất cả các chuỗi hoàn thành được tạo ra (bao gồm cả những chuỗi không kết thúc bằng `token` EOS).
  * `completions/min_length`: Độ dài ngắn nhất trong số tất cả các chuỗi hoàn thành được tạo ra.
  * `completions/max_length`: Độ dài lớn nhất trong số tất cả các chuỗi hoàn thành được tạo ra.
  * `completions/clipped_ratio`: Tỷ lệ các chuỗi hoàn thành không kết thúc bằng `token` EOS trước khi đạt đến độ dài tạo tối đa (tức là chúng đã bị cắt ngắn).
  * `completions/mean_terminated_length`: Độ dài trung bình chỉ của những chuỗi hoàn thành đã kết thúc thành công bằng `token` EOS.
  * `completions/min_terminated_length`: Độ dài ngắn nhất trong số các chuỗi hoàn thành kết thúc bằng `token` EOS.
  * `completions/max_terminated_length`: Độ dài lớn nhất trong số các chuỗi hoàn thành kết thúc bằng `token` EOS.

**Rewards (Phần thưởng):**

  * `rewards/{reward_func_name}/mean`: Phần thưởng trung bình thu được từ một hàm phần thưởng (reward function) cụ thể, có tên (ví dụ: `rewards/my_custom_reward/mean`). Chỉ số này được ghi lại cho mỗi hàm phần thưởng được sử dụng.
  * `rewards/{reward_func_name}/std`: Độ lệch chuẩn của phần thưởng từ một hàm phần thưởng cụ thể, có tên.
  * `reward`: Giá trị trung bình tổng thể của các phần thưởng (có thể đã được áp dụng trọng số và, nếu `args.scale_rewards` là `true`, đã được chuẩn hóa), sau khi chuẩn hóa theo nhóm (group-wise normalization) cho `advantages`.
  * `reward_std`: Độ lệch chuẩn của các phần thưởng (có thể đã được áp dụng trọng số) *trước khi* chuẩn hóa theo nhóm cho `advantages`.

**Policy and Loss Metrics (Các chỉ số về Policy và Loss):**

  * `kl`: Phân kỳ Kullback-Leibler (KL) trung bình giữa `policy` hiện tại và `reference policy`. Chỉ số này chỉ được ghi lại nếu `beta` (hệ số KL trong `GRPOConfig`) khác không.
  * Nếu `Liger GRPOLoss` được sử dụng (`use_liger_loss: True` trong `GRPOConfig`):
      * `clip_ratio`: Tỷ lệ các cập nhật `policy` mà tỷ lệ xác suất đã bị cắt bớt theo giới hạn `epsilon` của `GRPO loss`.
  * Nếu `GRPOLoss` tiêu chuẩn được sử dụng (`use_liger_loss: False`):
      * `clip_ratio/low_mean`: Tỷ lệ trung bình các trường hợp mà tỷ lệ xác suất `r_t(θ)` bị cắt ở giới hạn dưới `1 - epsilon_low` (xảy ra khi `advantage` là âm và tỷ lệ thấp hơn giới hạn).
      * `clip_ratio/low_min`: Tỷ lệ quan sát được tối thiểu cho `clip_ratio/low_mean` trên các `batch`/tiến trình.
      * `clip_ratio/high_mean`: Tỷ lệ trung bình các trường hợp mà tỷ lệ xác suất `r_t(θ)` bị cắt ở giới hạn trên `1 + epsilon_high` (xảy ra khi `advantage` là dương và tỷ lệ cao hơn giới hạn).
      * `clip_ratio/high_max`: Tỷ lệ quan sát được tối đa cho `clip_ratio/high_mean` trên các `batch`/tiến trình.
      * `clip_ratio/region_mean`: Tỷ lệ trung bình các trường hợp mà tỷ lệ xác suất bị cắt ở giới hạn dưới hoặc giới hạn trên.

### Các giá trị GRPO quan trọng

Trong quá trình huấn luyện GRPO, hãy theo dõi các giá trị này để có cái nhìn sâu sắc về hiệu suất và sự ổn định:

1.  `reward`: Đây là mục tiêu chính. Nó phản ánh các phần thưởng (đã được chuẩn hóa theo nhóm) mà `policy` đang đạt được. Nó thường sẽ tăng lên trong một quá trình huấn luyện thành công.
2.  `kl`: Nếu `beta > 0`, chỉ số này theo dõi sự phân kỳ so với mô hình tham chiếu. Hãy để mắt đến nó để đảm bảo `policy` không đi chệch quá xa, điều này có thể dẫn đến mất ổn định.
3.  `clip_ratio/*` (hoặc là `clip_ratio` cho Liger loss hoặc các chỉ số `clip_ratio/...` chi tiết hơn cho standard loss): Các chỉ số này cho biết tần suất các cập nhật `policy` bị giới hạn bởi cơ chế cắt bớt của GRPO. Giá trị rất cao có thể cho thấy `policy` đang cố gắng thay đổi quá đột ngột (có thể do `advantages` lớn hoặc `learning rate` quá cao) hoặc phạm vi cắt `epsilon` quá hạn chế.
4.  `completions/clipped_ratio`: Tỷ lệ cao ở đây cho thấy mô hình thường xuyên tạo ra các chuỗi hoàn thành bị cắt bởi `max_completion_length` thay vì kết thúc một cách tự nhiên bằng `token` EOS. Điều này có thể cho thấy các vấn đề trong việc học cách kết thúc chuỗi hoặc `max_completion_length` quá ngắn.
5.  `rewards/{reward_func_name}/mean`: Theo dõi giá trị trung bình của các hàm phần thưởng riêng lẻ có thể giúp chẩn đoán khía cạnh nào của hành vi mong muốn mà mô hình đang học tốt hoặc đang gặp khó khăn, đặc biệt khi sử dụng nhiều nguồn phần thưởng.