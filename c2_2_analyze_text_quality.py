import spacy
from collections import Counter

# uv add pip
# uv run python -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

def analyze_text_quality(text):
    doc = nlp(text)
    
    # Kiểm tra từ ngoài vocabulary (có thể là từ viết sai hoặc từ hiếm)
    oov_words = [token.text for token in doc if token.is_oov and token.is_alpha]
    
    # Đếm các loại từ loại để đánh giá grammar score
    pos_counts = Counter(token.pos_ for token in doc)
    grammar_score = pos_counts['NOUN'] + pos_counts['VERB'] + pos_counts['ADJ'] + pos_counts['ADV']
    
    # Kiểm tra câu không hoàn chỉnh (ít hơn 3 token hoặc không có động từ)
    incomplete_sentences = []
    for sent in doc.sents:
        tokens = [token for token in sent if not token.is_punct and not token.is_space]
        has_verb = any(token.pos_ == 'VERB' for token in tokens)
        if len(tokens) < 3 or not has_verb:
            incomplete_sentences.append(sent.text.strip())
    
    # Thống kê bổ sung về chất lượng văn bản
    word_count = len([token for token in doc if token.is_alpha])
    sentence_count = len(list(doc.sents))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    
    return {
        "oov_words": oov_words,  # Từ ngoài vocabulary (có thể là từ sai chính tả)
        "grammar_score": grammar_score,  # Điểm ngữ pháp dựa trên số lượng từ loại chính
        "incomplete_sentences": incomplete_sentences,  # Câu không hoàn chỉnh
        "word_count": word_count,  # Tổng số từ
        "sentence_count": sentence_count,  # Tổng số câu
        "avg_sentence_length": round(avg_sentence_length, 2)  # Độ dài câu trung bình
    }

# Ví dụ sử dụng
text = "This iz a smple txt with sum issues. Incomplet"
quality_report = analyze_text_quality(text)
print("Báo cáo chất lượng văn bản:")
for key, value in quality_report.items():
    print(f"- {key}: {value}")