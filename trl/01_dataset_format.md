Chắc chắn rồi, đây là bản dịch tài liệu sang tiếng Việt, tuân thủ các yêu cầu của bạn.

-----

# Các định dạng và loại Dataset

Hướng dẫn này cung cấp tổng quan về các định dạng và loại `dataset` được hỗ trợ bởi mỗi `trainer` trong `TRL`.

## Tổng quan về các định dạng và loại dataset

  - ***Định dạng*** (`format`) của một `dataset` đề cập đến cách dữ liệu được cấu trúc, thường được phân loại là *standard* (tiêu chuẩn) hoặc *conversational* (hội thoại).
  - ***Loại*** (`type`) được liên kết với tác vụ cụ thể mà `dataset` được thiết kế, chẳng hạn như *prompt-only* (chỉ có prompt) hoặc *preference* (sở thích). Mỗi loại được đặc trưng bởi các cột của nó, các cột này thay đổi tùy theo tác vụ, như được trình bày trong bảng.


### Định dạng

#### Standard (Tiêu chuẩn)

Định dạng `dataset` tiêu chuẩn thường bao gồm các chuỗi văn bản thuần túy. Các cột trong `dataset` thay đổi tùy thuộc vào tác vụ. Đây là định dạng được các `trainer` của `TRL` mong đợi. Dưới đây là các ví dụ về định dạng `dataset` tiêu chuẩn cho các tác vụ khác nhau:

```python
# Mô hình hóa ngôn ngữ (Language modeling)
language_modeling_example = {"text": "The sky is blue."}
# Sở thích (Preference)
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Sở thích không theo cặp (Unpaired preference)
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
```

#### Conversational (Hội thoại)

Các `dataset` hội thoại được sử dụng cho các tác vụ liên quan đến đối thoại hoặc tương tác trò chuyện giữa người dùng và trợ lý. Không giống như các định dạng `dataset` tiêu chuẩn, chúng chứa các chuỗi tin nhắn, trong đó mỗi tin nhắn có một `role` (vai trò, ví dụ: `"user"` hoặc `"assistant"`) và `content` (nội dung, tức văn bản tin nhắn).

```python
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
```

Cũng giống như các `dataset` tiêu chuẩn, các cột trong `dataset` hội thoại thay đổi tùy thuộc vào tác vụ. Dưới đây là các ví dụ về định dạng `dataset` hội thoại cho các tác vụ khác nhau:

```python
# Prompt và completion (Prompt-completion)
prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                             "completion": [{"role": "assistant", "content": "It is blue."}]}
# Sở thích (Preference)
preference_example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "chosen": [{"role": "assistant", "content": "It is blue."}],
    "rejected": [{"role": "assistant", "content": "It is green."}],
}
```

Các `dataset` hội thoại rất hữu ích để huấn luyện các mô hình trò chuyện (`chat models`), nhưng phải được chuyển đổi sang định dạng `standard` trước khi sử dụng với các `trainer` của `TRL`. Điều này thường được thực hiện bằng cách sử dụng các `chat templates` dành riêng cho mô hình đang được sử dụng. Để biết thêm thông tin, hãy tham khảo phần [Làm việc với dataset hội thoại trong TRL](https://www.google.com/search?q=%23l%C3%A0m-vi%E1%BB%87c-v%E1%BB%9Bi-dataset-h%E1%BB%99i-tho%E1%BA%A1i-trong-trl).

## Tool Calling (Gọi công cụ)

Một số `chat templates` hỗ trợ *tool calling*, cho phép mô hình tương tác với các hàm bên ngoài—được gọi là **tools** (công cụ)—trong quá trình sinh văn bản. Điều này mở rộng khả năng hội thoại của mô hình bằng cách cho phép nó xuất ra một trường `"tool_calls"` thay vì một tin nhắn `"content"` tiêu chuẩn mỗi khi nó quyết định gọi một công cụ.

Sau khi trợ lý khởi tạo một lệnh gọi công cụ, công cụ sẽ thực thi và trả về kết quả. Trợ lý sau đó có thể xử lý kết quả này và tiếp tục cuộc hội thoại một cách phù hợp.

Đây là một ví dụ đơn giản về tương tác gọi công cụ:

```python
messages = [
    {"role": "user", "content": "Turn on the living room lights."},
    {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {
            "name": "control_light",
            "arguments": {"room": "living room", "state": "on"}
        }}]
    },
    {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
    {"role": "assistant", "content": "Done!"}
]
```

Khi chuẩn bị `dataset` cho Supervised Fine-Tuning (`SFT`) với `tool calling`, điều quan trọng là `dataset` của bạn phải bao gồm một cột bổ sung có tên là `tools`. Cột này chứa danh sách các công cụ có sẵn cho mô hình, thường được `chat template` sử dụng để xây dựng `system prompt`.

Các công cụ phải được chỉ định ở định dạng `JSON schema` đã được mã hóa. Bạn có thể tự động tạo `schema` này từ chữ ký hàm Python bằng cách sử dụng tiện ích [`~transformers.utils.get_json_schema`]:

```python
from transformers.utils import get_json_schema

def control_light(room: str, state: str) -> str:
    """
    Controls the lights in a room.

    Args:
        room: The name of the room.
        state: The desired state of the light ("on" or "off").

    Returns:
        str: A message indicating the new state of the lights.
    """
    return f"The lights in {room} are now {state}."

# Generate JSON schema
json_schema = get_json_schema(control_light)
```

`Schema` được tạo ra sẽ trông như sau:

```python
{
    "type": "function",
    "function": {
        "name": "control_light",
        "description": "Controls the lights in a room.",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "description": "The name of the room."},
                "state": {"type": "string", "description": 'The desired state of the light ("on" or "off").'},
            },
            "required": ["room", "state"],
        },
        "return": {"type": "string", "description": "str: A message indicating the new state of the lights."},
    },
}
```

Một mục `dataset` hoàn chỉnh cho `SFT` có thể trông như sau:

```python
{"messages": messages, "tools": [json_schema]}
```

Để biết thêm thông tin chi tiết về `tool calling`, hãy tham khảo [mục Tool Calling trong tài liệu của `transformers`](https://www.google.com/search?q=%5Bhttps://huggingface.co/docs/transformers/chat_extras%23tools-and-rag%5D\(https://huggingface.co/docs/transformers/chat_extras%23tools-and-rag\)) và bài đăng blog [Tool Use, Unified](https://huggingface.co/blog/unified-tool-use).

### Các loại

#### Mô hình hóa ngôn ngữ (Language modeling)

Một `dataset` mô hình hóa ngôn ngữ bao gồm một cột `"text"` (hoặc `"messages"` cho các `dataset` hội thoại) chứa một chuỗi văn bản hoàn chỉnh.

```python
# Định dạng standard
language_modeling_example = {"text": "The sky is blue."}
# Định dạng conversational
language_modeling_example = {"messages": [
    {"role": "user", "content": "What color is the sky?"},
    {"role": "assistant", "content": "It is blue."}
]}
```

#### Chỉ có prompt (Prompt-only)

Trong một `dataset` chỉ có `prompt`, chỉ có `prompt` ban đầu (câu hỏi hoặc câu chưa hoàn chỉnh) được cung cấp dưới khóa `"prompt"`. Quá trình huấn luyện thường bao gồm việc tạo ra `completion` dựa trên `prompt` này, nơi mô hình học cách tiếp tục hoặc hoàn thành đầu vào đã cho.

```python
# Định dạng standard
prompt_only_example = {"prompt": "The sky is"}
# Định dạng conversational
prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
```

Để xem các ví dụ về `dataset` chỉ có `prompt`, hãy tham khảo [bộ sưu tập Prompt-only datasets](https://huggingface.co/collections/trl-lib/prompt-only-datasets-677ea25245d20252cea00368).

\<Tip\>

Mặc dù cả hai loại `prompt-only` và `language modeling` đều tương tự nhau, chúng khác nhau ở cách xử lý đầu vào. Trong loại `prompt-only`, `prompt` đại diện cho một đầu vào chưa hoàn chỉnh mà mong đợi mô hình sẽ hoàn thành hoặc tiếp tục, trong khi ở loại `language modeling`, đầu vào được coi là một câu hoặc chuỗi hoàn chỉnh. Hai loại này được `TRL` xử lý khác nhau. Dưới đây là một ví dụ cho thấy sự khác biệt trong đầu ra của hàm `apply_chat_template` cho mỗi loại:

```python
from transformers import AutoTokenizer
from trl import apply_chat_template

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# Ví dụ cho loại prompt-only
prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
apply_chat_template(prompt_only_example, tokenizer)
# Đầu ra: {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n'}

# Ví dụ cho loại language modeling
lm_example = {"messages": [{"role": "user", "content": "What color is the sky?"}]}
apply_chat_template(lm_example, tokenizer)
# Đầu ra: {'text': '<|user|>\nWhat color is the sky?<|end|>\n<|endoftext|>'}
```

  - Đầu ra của `prompt-only` bao gồm một `'<|assistant|>\n'`, cho biết sự bắt đầu lượt của trợ lý và mong đợi mô hình tạo ra một `completion`.
  - Ngược lại, đầu ra của `language modeling` coi đầu vào là một chuỗi hoàn chỉnh và kết thúc nó bằng `'<|endoftext|>'`, báo hiệu sự kết thúc của văn bản và không mong đợi bất kỳ nội dung bổ sung nào.

\</Tip\>

#### Prompt và completion (Prompt-completion)

Một `dataset` `prompt-completion` bao gồm một `"prompt"` và một `"completion"`.

```python
# Định dạng standard
prompt_completion_example = {"prompt": "The sky is", "completion": " blue."}
# Định dạng conversational
prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                             "completion": [{"role": "assistant", "content": "It is blue."}]}
```

Để xem các ví dụ về `dataset` `prompt-completion`, hãy tham khảo [bộ sưu tập Prompt-completion datasets](https://huggingface.co/collections/trl-lib/prompt-completion-datasets-677ea2bb20bbb6bdccada216).

#### Sở thích (Preference)

Một `dataset` sở thích (`preference`) được sử dụng cho các tác vụ mà mô hình được huấn luyện để chọn giữa hai hoặc nhiều `completion` có thể có cho cùng một `prompt`. `Dataset` này bao gồm một `"prompt"`, một `completion` `"chosen"` (được chọn), và một `completion` `"rejected"` (bị từ chối). Mô hình được huấn luyện để chọn câu trả lời `"chosen"` thay vì câu trả lời `"rejected"`.
Một số `dataset` có thể không bao gồm cột `"prompt"`, trong trường hợp đó `prompt` là ngầm định và được bao gồm trực tiếp trong các `completion` `"chosen"` và `"rejected"`. Chúng tôi khuyên bạn nên sử dụng `prompt` tường minh bất cứ khi nào có thể.

```python
# Định dạng standard
## Prompt tường minh (khuyến nghị)
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Prompt ngầm định
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}

# Định dạng conversational
## Prompt tường minh (khuyến nghị)
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]}
## Prompt ngầm định
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]}
```

Để xem các ví dụ về `dataset` sở thích, hãy tham khảo [bộ sưu tập Preference datasets](https://huggingface.co/collections/trl-lib/preference-datasets-677e99b581018fcad9abd82c).

Một số `dataset` sở thích có thể được tìm thấy với [thẻ `dpo` trên Hugging Face Hub](https://www.google.com/search?q=%5Bhttps://huggingface.co/datasets%3Fother%3Ddpo%5D\(https://huggingface.co/datasets%3Fother%3Ddpo\)). Bạn cũng có thể khám phá [DPO Collections của librarian-bots](https://huggingface.co/collections/librarian-bots/direct-preference-optimization-datasets-66964b12835f46289b6ef2fc) để xác định các `dataset` sở thích.

#### Sở thích không theo cặp (Unpaired preference)

Một `dataset` sở thích không theo cặp (`unpaired preference`) tương tự như một `dataset` sở thích nhưng thay vì có các `completion` `"chosen"` và `"rejected"` cho cùng một `prompt`, nó bao gồm một `"completion"` duy nhất và một `"label"` cho biết `completion` đó có được ưa thích hay không.

```python
# Định dạng standard
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
# Định dạng conversational
unpaired_preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                               "completion": [{"role": "assistant", "content": "It is blue."}],
                               "label": True}
```

Để xem các ví dụ về `dataset` sở thích không theo cặp, hãy tham khảo [bộ sưu tập Unpaired preference datasets](https://huggingface.co/collections/trl-lib/unpaired-preference-datasets-677ea22bf5f528c125b0bcdf).

#### Giám sát theo từng bước (Stepwise supervision)

Một `dataset` giám sát theo từng bước (hoặc quy trình) tương tự như một `dataset` [sở thích không theo cặp](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p-unpaired-preference) nhưng bao gồm nhiều bước của `completion`, mỗi bước có nhãn riêng. Cấu trúc này hữu ích cho các tác vụ cần ghi nhãn chi tiết, từng bước, chẳng hạn như các tác vụ suy luận. Bằng cách đánh giá từng bước riêng biệt và cung cấp các nhãn mục tiêu, phương pháp này giúp xác định chính xác nơi suy luận đúng và nơi xảy ra lỗi, cho phép phản hồi có mục tiêu trên từng phần của quá trình suy luận.

```python
stepwise_example = {
    "prompt": "Which number is larger, 9.8 or 9.11?",
    "completions": ["The fractional part of 9.8 is 0.8, while the fractional part of 9.11 is 0.11.", "Since 0.11 is greater than 0.8, the number 9.11 is larger than 9.8."],
    "labels": [True, False]
}
```

Để xem các ví dụ về `dataset` giám sát theo từng bước, hãy tham khảo [bộ sưu tập Stepwise supervision datasets](https://huggingface.co/collections/trl-lib/stepwise-supervision-datasets-677ea27fd4c5941beed7a96e).

## Nên sử dụng loại dataset nào?

Việc chọn đúng loại `dataset` phụ thuộc vào tác vụ bạn đang thực hiện và các yêu cầu cụ thể của `trainer` `TRL` bạn đang sử dụng. Dưới đây là tổng quan ngắn gọn về các loại `dataset` được hỗ trợ bởi mỗi `trainer` `TRL`.

| Trainer                 | Loại dataset mong đợi                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------ |
| [`BCOTrainer`]          | [Sở thích không theo cặp](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p-unpaired-preference)                                                            |
| [`CPOTrainer`]          | [Sở thích (khuyến nghị prompt tường minh)](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-preference)                                                |
| [`DPOTrainer`]          | [Sở thích (khuyến nghị prompt tường minh)](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-preference)                                                |
| [`GKDTrainer`]          | [Prompt và completion](https://www.google.com/search?q=%23prompt-v%C3%A0-completion-prompt-completion)                                                    |
| [`GRPOTrainer`]         | [Chỉ có prompt](https://www.google.com/search?q=%23ch%E1%BB%89-c%C3%B3-prompt-prompt-only)                                                                            |
| [`IterativeSFTTrainer`] | [Sở thích không theo cặp](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p-unpaired-preference)                                                            |
| [`KTOTrainer`]          | [Sở thích không theo cặp](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p-unpaired-preference) hoặc [Sở thích (khuyến nghị prompt tường minh)](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-preference) |
| [`NashMDTrainer`]       | [Chỉ có prompt](https://www.google.com/search?q=%23ch%E1%BB%89-c%C3%B3-prompt-prompt-only)                                                                            |
| [`OnlineDPOTrainer`]    | [Chỉ có prompt](https://www.google.com/search?q=%23ch%E1%BB%89-c%C3%B3-prompt-prompt-only)                                                                            |
| [`ORPOTrainer`]         | [Sở thích (khuyến nghị prompt tường minh)](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-preference)                                                |
| [`PPOTrainer`]          | Mô hình hóa ngôn ngữ đã được token hóa (`Tokenized language modeling`)                                                                            |
| [`PRMTrainer`]          | [Giám sát theo từng bước](https://www.google.com/search?q=%23gi%C3%A1m-s%C3%A1t-theo-t%E1%BB%ABng-b%C6%B0%E1%BB%9Bc-stepwise-supervision)                                                          |
| [`RewardTrainer`]       | [Sở thích (khuyến nghị prompt ngầm định)](https://www.google.com/search?q=%23s%E1%BB%9F-th%C3%ADch-preference)                                                |
| [`SFTTrainer`]          | [Mô hình hóa ngôn ngữ](https://www.google.com/search?q=%23m%C3%B4-h%C3%ACnh-h%C3%B3a-ng%C3%B4n-ng%E1%BB%AF-language-modeling) hoặc [Prompt và completion](https://www.google.com/search?q=%23prompt-v%C3%A0-completion-prompt-completion)                     |
| [`XPOTrainer`]          | [Chỉ có prompt](https://www.google.com/search?q=%23ch%E1%BB%89-c%C3%B3-prompt-prompt-only)                                                                            |

\<Tip\>

Các `trainer` của `TRL` chỉ hỗ trợ các định dạng `dataset` tiêu chuẩn, [tính đến thời điểm hiện tại](https://github.com/huggingface/trl/issues/2071). Nếu bạn có một `dataset` hội thoại, trước tiên bạn phải chuyển đổi nó sang định dạng tiêu chuẩn.
Để biết thêm thông tin về cách làm việc với `dataset` hội thoại, hãy tham khảo phần [Làm việc với dataset hội thoại trong TRL](https://www.google.com/search?q=%23l%C3%A0m-vi%E1%BB%87c-v%E1%BB%9Bi-dataset-h%E1%BB%99i-tho%E1%BA%A1i-trong-trl).

\</Tip\>

## Làm việc với dataset hội thoại trong TRL

Các `dataset` hội thoại ngày càng phổ biến, đặc biệt là để huấn luyện các mô hình trò chuyện (`chat models`). Tuy nhiên, một số `trainer` của `TRL` không hỗ trợ `dataset` hội thoại ở định dạng thô của chúng. (Để biết thêm thông tin, xem [issue \#2071](https://github.com/huggingface/trl/issues/2071).) Các `dataset` này trước tiên phải được chuyển đổi sang định dạng tiêu chuẩn.
May mắn thay, `TRL` cung cấp các công cụ để dễ dàng xử lý việc chuyển đổi này, được trình bày chi tiết dưới đây.

### Chuyển đổi một dataset hội thoại thành một dataset tiêu chuẩn

Để chuyển đổi một `dataset` hội thoại thành một `dataset` tiêu chuẩn, bạn cần *áp dụng một chat template* cho `dataset`. Một `chat template` là một cấu trúc được xác định trước thường bao gồm các trình giữ chỗ cho tin nhắn của người dùng và trợ lý. `Template` này được cung cấp bởi `tokenizer` của mô hình bạn sử dụng.

Để biết hướng dẫn chi tiết về cách sử dụng `chat templating`, hãy tham khảo [mục Chat templating trong tài liệu của `transformers`](https://www.google.com/search?q=%5Bhttps://huggingface.co/docs/transformers/en/chat_templating%5D\(https://huggingface.co/docs/transformers/en/chat_templating\)).

Trong `TRL`, phương thức bạn áp dụng để chuyển đổi `dataset` sẽ thay đổi tùy thuộc vào tác vụ. May mắn thay, `TRL` cung cấp một hàm trợ giúp có tên là [`apply_chat_template`] để đơn giản hóa quá trình này. Đây là một ví dụ về cách sử dụng nó:

```python
from transformers import AutoTokenizer
from trl import apply_chat_template

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "completion": [{"role": "assistant", "content": "It is blue."}]
}

apply_chat_template(example, tokenizer)
# Đầu ra:
# {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n', 'completion': 'It is blue.<|end|>\n<|endoftext|>'}
```

Ngoài ra, bạn có thể sử dụng phương thức [`~datasets.Dataset.map`] để áp dụng `template` trên toàn bộ `dataset`:

```python
from datasets import Dataset
from trl import apply_chat_template

dataset_dict = {
    "prompt": [[{"role": "user", "content": "What color is the sky?"}],
               [{"role": "user", "content": "Where is the sun?"}]],
    "completion": [[{"role": "assistant", "content": "It is blue."}],
                   [{"role": "assistant", "content": "In the sky."}]]
}

dataset = Dataset.from_dict(dataset_dict)
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
# Đầu ra:
# {'prompt': ['<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n',
#             '<|user|>\nWhere is the sun?<|end|>\n<|assistant|>\n'],
#  'completion': ['It is blue.<|end|>\n<|endoftext|>', 'In the sky.<|end|>\n<|endoftext|>']}
```

\<Tip warning={true}\>

Chúng tôi khuyên bạn nên sử dụng hàm [`apply_chat_template`] thay vì gọi trực tiếp `tokenizer.apply_chat_template`. Việc xử lý các `chat template` cho các `dataset` không phải là `language modeling` có thể phức tạp và có thể dẫn đến lỗi, chẳng hạn như đặt nhầm `system prompt` vào giữa một cuộc hội thoại.
Để biết thêm ví dụ, xem [\#1930 (comment)](https://github.com/huggingface/trl/pull/1930#issuecomment-2292908614). Hàm [`apply_chat_template`] được thiết kế để xử lý những sự phức tạp này và đảm bảo áp dụng đúng các `chat template` cho các tác vụ khác nhau.

\</Tip\>

\<Tip warning={true}\>

Điều quan trọng cần lưu ý là các `chat template` là đặc trưng cho từng mô hình. Ví dụ, nếu bạn sử dụng `chat template` từ [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) với ví dụ trên, bạn sẽ nhận được một đầu ra khác:

```python
apply_chat_template(example, AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct"))
# Đầu ra:
# {'prompt': '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat color is the sky?<|im_end|>\n<|im_start|>assistant\n',
#  'completion': 'It is blue.<|im_end|>\n'}
```

Luôn sử dụng `chat template` được liên kết với mô hình bạn đang làm việc. Sử dụng sai `template` có thể dẫn đến kết quả không chính xác hoặc không mong muốn.

\</Tip\>

## Sử dụng bất kỳ dataset nào với TRL: tiền xử lý và chuyển đổi

Nhiều `dataset` có định dạng được thiết kế riêng cho các tác vụ cụ thể, có thể không tương thích trực tiếp với `TRL`. Để sử dụng các `dataset` như vậy với `TRL`, bạn có thể cần phải tiền xử lý và chuyển đổi chúng sang định dạng bắt buộc.

Để làm điều này dễ dàng hơn, chúng tôi cung cấp một bộ [script ví dụ](https://github.com/huggingface/trl/tree/main/examples/datasets) bao gồm các chuyển đổi `dataset` phổ biến.

### Ví dụ: Dataset UltraFeedback

Hãy lấy [dataset UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) làm ví dụ. Dưới đây là bản xem trước của `dataset`:

\<iframe
  src="[https://huggingface.co/datasets/openbmb/UltraFeedback/embed/viewer/default/train](https://huggingface.co/datasets/openbmb/UltraFeedback/embed/viewer/default/train)"
  frameborder="0"
  width="100%"
  height="560px"
\>\</iframe\>

Như được hiển thị ở trên, định dạng `dataset` không khớp với cấu trúc mong đợi. Nó không ở định dạng hội thoại, tên cột khác nhau và kết quả liên quan đến các mô hình khác nhau (ví dụ: Bard, GPT-4) và các khía cạnh (ví dụ: "helpfulness", "honesty").

Bằng cách sử dụng script chuyển đổi được cung cấp [`examples/datasets/ultrafeedback.py`](https://www.google.com/search?q=%5Bhttps://github.com/huggingface/trl/tree/main/examples/datasets/ultrafeedback.py%5D\(https://github.com/huggingface/trl/tree/main/examples/datasets/ultrafeedback.py\)), bạn có thể chuyển đổi `dataset` này thành loại sở thích không theo cặp (`unpaired preference`), và đẩy nó lên Hub:

```sh
python examples/datasets/ultrafeedback.py --push_to_hub --repo_id trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness
```

Sau khi chuyển đổi, `dataset` sẽ trông như thế này:

\<iframe
  src="[https://huggingface.co/datasets/trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness/embed/viewer/default/train?row=0](https://huggingface.co/datasets/trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness/embed/viewer/default/train?row=0)"
  frameborder="0"
  width="100%"
  height="560px"
\>\</iframe\>

Bây giờ, bạn có thể sử dụng `dataset` này với `TRL`\!

Bằng cách điều chỉnh các script được cung cấp hoặc tạo script của riêng bạn, bạn có thể chuyển đổi bất kỳ `dataset` nào thành một định dạng tương thích với `TRL`.

## Các tiện ích để chuyển đổi các loại dataset

Phần này cung cấp mã ví dụ để giúp bạn chuyển đổi giữa các loại `dataset` khác nhau. Mặc dù một số chuyển đổi có thể được thực hiện sau khi áp dụng `chat template` (tức là ở định dạng `standard`), chúng tôi khuyên bạn nên thực hiện chuyển đổi trước khi áp dụng `chat template` để đảm bảo nó hoạt động nhất quán.

Để đơn giản, một số ví dụ dưới đây không tuân theo khuyến nghị này và sử dụng định dạng `standard`. Tuy nhiên, các chuyển đổi có thể được áp dụng trực tiếp cho định dạng hội thoại mà không cần sửa đổi.

| Từ \\ Đến                       | Mô hình hóa ngôn ngữ                                                       | Prompt-completion                                                       | Prompt-only                                                       | Sở thích với prompt ngầm định                           | Sở thích                                                | Sở thích không theo cặp                                                       | Giám sát theo từng bước |
| ------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------- | -------------------- |
| Mô hình hóa ngôn ngữ               | N/A                                                                     | N/A                                                                     | N/A                                                               | N/A                                                       | N/A                                                       | N/A                                                                       | N/A                  |
| Prompt-completion               | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-prompt-completion-sang-language-modeling)               | N/A                                                                     | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-prompt-completion-sang-prompt-only)               | N/A                                                       | N/A                                                       | N/A                                                                       | N/A                  |
| Prompt-only                     | N/A                                                                     | N/A                                                                     | N/A                                                               | N/A                                                       | N/A                                                       | N/A                                                                       | N/A                  |
| Sở thích với prompt ngầm định | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-v%E1%BB%9Bi-prompt-ng%E1%BA%A7m-%C4%91%E1%BB%8Bnh-sang-language-modeling) | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-v%E1%BB%9Bi-prompt-ng%E1%BA%A7m-%C4%91%E1%BB%8Bnh-sang-prompt-completion) | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-v%E1%BB%9Bi-prompt-ng%E1%BA%A7m-%C4%91%E1%BB%8Bnh-sang-prompt-only) | N/A                                                       | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-v%E1%BB%9Bi-prompt-ng%E1%BA%A7m-%C4%91%E1%BB%8Bnh-sang-t%C6%B0%E1%BB%9Dng-minh) | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-v%E1%BB%9Bi-prompt-ng%E1%BA%A7m-%C4%91%E1%BB%8Bnh-sang-s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p) | N/A                  |
| Sở thích                      | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-sang-language-modeling)                      | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-sang-prompt-completion)                      | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-sang-prompt-only)                      | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-v%E1%BB%9Bi-prompt-t%C6%B0%E1%BB%9Dng-minh-sang-ng%E1%BA%A7m-%C4%91%E1%BB%8Bnh) | N/A                                                       | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-sang-s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p)                      | N/A                  |
| Sở thích không theo cặp             | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p-sang-language-modeling)             | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p-sang-prompt-completion)             | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p-sang-prompt-only)             | N/A                                                       | N/A                                                       | N/A                                                                       | N/A                  |
| Giám sát theo từng bước            | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-gi%C3%A1m-s%C3%A1t-theo-t%E1%BB%ABng-b%C6%B0%E1%BB%9Bc-sang-language-modeling)            | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-gi%C3%A1m-s%C3%A1t-theo-t%E1%BB%ABng-b%C6%B0%E1%BB%9Bc-sang-prompt-completion)            | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-gi%C3%A1m-s%C3%A1t-theo-t%E1%BB%ABng-b%C6%B0%E1%BB%9Bc-sang-prompt-only)            | N/A                                                       | N/A                                                       | [🔗](https://www.google.com/search?q=%23t%E1%BB%AB-dataset-gi%C3%A1m-s%C3%A1t-theo-t%E1%BB%ABng-b%C6%B0%E1%BB%9Bc-sang-s%E1%BB%9F-th%C3%ADch-kh%C3%B4ng-theo-c%E1%BA%B7p)            | N/A                  |

### Từ dataset prompt-completion sang language modeling

Để chuyển đổi một `dataset` `prompt-completion` thành một `dataset` `language modeling`, hãy nối `prompt` và `completion`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "completion": [" blue.", " in the sky."],
})

def concat_prompt_completion(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.map(concat_prompt_completion, remove_columns=["prompt", "completion"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### Từ dataset prompt-completion sang prompt-only

Để chuyển đổi một `dataset` `prompt-completion` thành một `dataset` `prompt-only`, hãy xóa `completion`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "completion": [" blue.", " in the sky."],
})

dataset = dataset.remove_columns("completion")
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### Từ dataset sở thích với prompt ngầm định sang language modeling

Để chuyển đổi một `dataset` sở thích với `prompt` ngầm định thành một `dataset` `language modeling`, hãy xóa cột `rejected` và đổi tên cột `"chosen"` thành `"text"`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "chosen": ["The sky is blue.", "The sun is in the sky."],
    "rejected": ["The sky is green.", "The sun is in the sea."],
})

dataset = dataset.rename_column("chosen", "text").remove_columns("rejected")
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### Từ dataset sở thích với prompt ngầm định sang prompt-completion

Để chuyển đổi một `dataset` sở thích với `prompt` ngầm định thành một `dataset` `prompt-completion`, hãy trích xuất `prompt` bằng [`extract_prompt`], xóa cột `rejected`, và đổi tên cột `"chosen"` thành `"completion"`.

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})
dataset = dataset.map(extract_prompt).remove_columns("rejected").rename_column("chosen", "completion")
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}], 'completion': [{'role': 'assistant', 'content': 'It is blue.'}]}
```

### Từ dataset sở thích với prompt ngầm định sang prompt-only

Để chuyển đổi một `dataset` sở thích với `prompt` ngầm định thành một `dataset` `prompt-only`, hãy trích xuất `prompt` bằng [`extract_prompt`], và xóa các cột `rejected` và `chosen`.

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})
dataset = dataset.map(extract_prompt).remove_columns(["chosen", "rejected"])
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}]}
```

### Từ dataset sở thích với prompt ngầm định sang tường minh

Để chuyển đổi một `dataset` sở thích với `prompt` ngầm định thành một `dataset` sở thích với `prompt` tường minh, hãy trích xuất `prompt` bằng [`extract_prompt`].

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = dataset.map(extract_prompt)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
```

### Từ dataset sở thích với prompt ngầm định sang sở thích không theo cặp

Để chuyển đổi một `dataset` sở thích với `prompt` ngầm định thành một `dataset` sở thích không theo cặp, hãy trích xuất `prompt` bằng [`extract_prompt`], và tách cặp `dataset` bằng [`unpair_preference_dataset`].

```python
from datasets import Dataset
from trl import extract_prompt, unpair_preference_dataset

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = dataset.map(extract_prompt)
dataset = unpair_preference_dataset(dataset)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'completion': [{'role': 'assistant', 'content': 'It is blue.'}],
 'label': True}
```

\<Tip warning={true}\>

Hãy nhớ rằng các `completion` `"chosen"` và `"rejected"` trong một `dataset` sở thích có thể là tốt hoặc xấu.
Trước khi áp dụng [`unpair_preference_dataset`], hãy đảm bảo rằng tất cả các `completion` `"chosen"` có thể được gán nhãn là tốt và tất cả các `completion` `"rejected"` là xấu.
Điều này có thể được đảm bảo bằng cách kiểm tra điểm đánh giá tuyệt đối của mỗi `completion`, ví dụ như từ một mô hình phần thưởng (`reward model`).

\</Tip\>

### Từ dataset sở thích sang language modeling

Để chuyển đổi một `dataset` sở thích thành một `dataset` `language modeling`, hãy xóa cột `rejected`, nối `prompt` và `chosen` vào cột `"text"`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

def concat_prompt_chosen(example):
    return {"text": example["prompt"] + example["chosen"]}

dataset = dataset.map(concat_prompt_chosen, remove_columns=["prompt", "chosen", "rejected"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### Từ dataset sở thích sang prompt-completion

Để chuyển đổi một `dataset` sở thích thành một `dataset` `prompt-completion`, hãy xóa cột `rejected`, và đổi tên cột `"chosen"` thành `"completion"`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

dataset = dataset.remove_columns("rejected").rename_column("chosen", "completion")
```

```python
>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.'}
```

### Từ dataset sở thích sang prompt-only

Để chuyển đổi một `dataset` sở thích thành một `dataset` `prompt-only`, hãy xóa các cột `rejected` và `chosen`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

dataset = dataset.remove_columns(["chosen", "rejected"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### Từ dataset sở thích với prompt tường minh sang ngầm định

Để chuyển đổi một `dataset` sở thích với `prompt` tường minh thành một `dataset` sở thích với `prompt` ngầm định, hãy nối `prompt` vào cả `chosen` và `rejected`, và xóa cột `prompt`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "chosen": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "assistant", "content": "It is green."}],
        [{"role": "assistant", "content": "In the sea."}],
    ],
})

def concat_prompt_to_completions(example):
    return {"chosen": example["prompt"] + example["chosen"], "rejected": example["prompt"] + example["rejected"]}

dataset = dataset.map(concat_prompt_to_completions, remove_columns="prompt")
```

```python
>>> dataset[0]
{'chosen': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is green.'}]}
```

### Từ dataset sở thích sang sở thích không theo cặp

Để chuyển đổi `dataset` thành một `dataset` sở thích không theo cặp, hãy tách cặp `dataset` bằng [`unpair_preference_dataset`].

```python
from datasets import Dataset
from trl import unpair_preference_dataset

dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "chosen": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "assistant", "content": "It is green."}],
        [{"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = unpair_preference_dataset(dataset)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'completion': [{'role': 'assistant', 'content': 'It is blue.'}],
 'label': True}
```

\<Tip warning={true}\>

Hãy nhớ rằng các `completion` `"chosen"` và `"rejected"` trong một `dataset` sở thích có thể là tốt hoặc xấu.
Trước khi áp dụng [`unpair_preference_dataset`], hãy đảm bảo rằng tất cả các `completion` `"chosen"` có thể được gán nhãn là tốt và tất cả các `completion` `"rejected"` là xấu.
Điều này có thể được đảm bảo bằng cách kiểm tra điểm đánh giá tuyệt đối của mỗi `completion`, ví dụ như từ một mô hình phần thưởng (`reward model`).

\</Tip\>

### Từ dataset sở thích không theo cặp sang language modeling

Để chuyển đổi một `dataset` sở thích không theo cặp thành một `dataset` `language modeling`, hãy nối các `prompt` với các `completion` tốt vào cột `"text"`, và xóa các cột `prompt`, `completion` và `label`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

def concatenate_prompt_completion(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.filter(lambda x: x["label"]).map(concatenate_prompt_completion).remove_columns(["prompt", "completion", "label"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### Từ dataset sở thích không theo cặp sang prompt-completion

Để chuyển đổi một `dataset` sở thích không theo cặp thành một `dataset` `prompt-completion`, hãy lọc các nhãn tốt, sau đó xóa các cột nhãn.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

dataset = dataset.filter(lambda x: x["label"]).remove_columns(["label"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.'}
```

### Từ dataset sở thích không theo cặp sang prompt-only

Để chuyển đổi một `dataset` sở thích không theo cặp thành một `dataset` `prompt-only`, hãy xóa các cột `completion` và `label`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

dataset = dataset.remove_columns(["completion", "label"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### Từ dataset giám sát theo từng bước sang language modeling

Để chuyển đổi một `dataset` giám sát theo từng bước thành một `dataset` `language modeling`, hãy nối các `prompt` với các `completion` tốt vào cột `"text"`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def concatenate_prompt_completions(example):
    completion = "".join(example["completions"])
    return {"text": example["prompt"] + completion}

dataset = dataset.filter(lambda x: all(x["labels"])).map(concatenate_prompt_completions, remove_columns=["prompt", "completions", "labels"])
```

```python
>>> dataset[0]
{'text': 'Blue light scatters more in the atmosphere, so the sky is green.'}
```

### Từ dataset giám sát theo từng bước sang prompt completion

Để chuyển đổi một `dataset` giám sát theo từng bước thành một `dataset` `prompt-completion`, hãy nối các `completion` tốt lại và xóa các nhãn.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def join_completions(example):
    completion = "".join(example["completions"])
    return {"completion": completion}

dataset = dataset.filter(lambda x: all(x["labels"])).map(join_completions, remove_columns=["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light', 'completion': ' scatters more in the atmosphere, so the sky is green.'}
```

### Từ dataset giám sát theo từng bước sang prompt only

Để chuyển đổi một `dataset` giám sát theo từng bước thành một `dataset` `prompt-only`, hãy xóa các cột `completions` và `labels`.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

dataset = dataset.remove_columns(["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light'}
```

### Từ dataset giám sát theo từng bước sang sở thích không theo cặp

Để chuyển đổi một `dataset` giám sát theo từng bước thành một `dataset` sở thích không theo cặp, hãy nối các `completions` và hợp nhất các `labels`.

Phương pháp hợp nhất các nhãn phụ thuộc vào tác vụ cụ thể. Trong ví dụ này, chúng tôi sử dụng phép toán AND logic. Điều này có nghĩa là nếu các nhãn của từng bước cho biết tính đúng đắn của các bước riêng lẻ, nhãn kết quả sẽ phản ánh tính đúng đắn của toàn bộ chuỗi.

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def merge_completions_and_labels(example):
    return {"prompt": example["prompt"], "completion": "".join(example["completions"]), "label": all(example["labels"])}

dataset = dataset.map(merge_completions_and_labels, remove_columns=["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light', 'completion': ' scatters more in the atmosphere, so the sky is green.', 'label': False}
```

## Vision datasets (Dataset hình ảnh)

Một số `trainer` cũng hỗ trợ tinh chỉnh các mô hình ngôn ngữ-thị giác (vision-language models - VLMs) bằng cách sử dụng các cặp hình ảnh-văn bản. Trong trường hợp này, khuyến nghị sử dụng định dạng hội thoại, vì mỗi mô hình xử lý các trình giữ chỗ hình ảnh trong văn bản theo cách khác nhau.

Một `dataset` thị giác hội thoại khác với một `dataset` hội thoại tiêu chuẩn ở hai điểm chính:

1.  `Dataset` phải chứa khóa `images` với dữ liệu hình ảnh.
2.  Trường `"content"` trong các tin nhắn phải là một danh sách các dictionary, trong đó mỗi dictionary chỉ định loại dữ liệu: `"image"` hoặc `"text"`.

Ví dụ:

```python
# Dataset văn bản:
"content": "What color is the sky?"

# Dataset thị giác:
"content": [
    {"type": "image"}, 
    {"type": "text", "text": "What color is the sky in the image?"}
]
```

Một ví dụ về `dataset` thị giác hội thoại là [openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset). Dưới đây là chế độ xem nhúng của dữ liệu huấn luyện của `dataset`, cho phép bạn khám phá nó trực tiếp:

\<iframe
  src="[https://huggingface.co/datasets/trl-lib/rlaif-v/embed/viewer/default/train](https://huggingface.co/datasets/trl-lib/rlaif-v/embed/viewer/default/train)"
  frameborder="0"
  width="100%"
  height="560px"
\>\</iframe\>