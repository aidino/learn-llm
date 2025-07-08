# ### **Tổng quan**

# Loại bỏ trùng lặp (Deduplication) là một bước quan trọng trong việc chuẩn bị csc kho văn bản lớn để huấn luyện Mô hình Ngôn ngữ Lớn (LLM)[cite: 371].
# Nội dung trùng lặp có thể dẫn đến việc mô hình bị thiên vị và lãng phí tài nguyên tính toán[cite: 371].
# Mục này giới thiệu các chiến lược khác nhau để xác định và loại bỏ dữ liệu trùng lặp một cách hiệu quả[cite: 371].

# ### **Các Chiến Lược Loại Bỏ Trùng Lặp Chính**

# Sách đề cmp đến bốn chiến lược chính để xử lý dữ liệu trùng lặp[cite: 371, 372]:

# 1.  **Loại bỏ trùng lặp hoàn toàn (Exact match deduplication)**
#     * **Giải thích**: Đây là phương pháp đơn giản nhất, nhằm mục đích loại bỏ các mẫu văn bản giống hệt nhau[cite: 371].
#     * **Ví dụ trong sech**: Nếu có một danh sách địa chỉ khách hàng và một địa chỉ xuất hiện nhiều lần một cách chính xác, 
# ví dụ "123 Main St, Anytown, CA 91234", thì các bản sao giống hệt sẽ bị xóa đi[cite: 375, 377].

# 2.  **Phát hiện trùng lặp gần đúng (Near-duplicate detection)**
#     * **Giải thích**: Phương pháp này xác định và loại bỏ các mẫu văn bản rất giống nhau về mặt nội dung, ngay cả khi câu chữ có chút khác biệt[cite: 372, 382].
#     * **Ví dụ trong sách**: Hai bài báo có thể báo cáo cùng một tin tức ("công ty báo cáo lợi nhuận hàng quý tăng đáng kể") 
# nhưng dùng cách diễn đạt hơi khác nhau. Thuật toán phát hiện trùng lặp gần đúng sẽ xác định chúng có nội dung tương tự và loại bỏ một trong hai[cite: 380, 381, 382].

# 3.  **Shingling**
#     * **Giải thích**: Kỹ thuật này tạo ra các chuỗi từ ngữ nhỏ, chồng chéo lên nhau (gọi là "shingles") để dùng cho việc so sánh[cite: 372]. 
# Một tài liệu sau đó được đại diện bởi một tập hợp các shingle này để so sánh với các tài liệu khác[cite: 390].
#     * **Ví dụ strong sách**: Đối với câu "The quick brown fox jumps over the lazy dog." và kích thước shingle là 3 từ (k=3), các shingle được tạo ra sẽ là: "The quick brown", "quick brown fox", "brown fox jumps", v.v...[cite: 384, 387, 388, 389, 390].

# 4.  **Băm nhạy cục bộ (Locality Sensitive Hashing - LSH)**
#     * **Giải thích**: LSH là một phương pháp hiệu quả để tìm các mục tương tự trong các tập dữ liệu lớn[cite: 372]. 
# Nó hoạt động bằng cách băm (hashing) các mục tương tự vào cùng một "bucket" (xô) với xác suất cao, giúp thu hẹp đáng kể phạm vi tìm kiếm[cite: 393, 394].
#     * **Ví dụ trong sách**: Với một cơ sở dữ liệu lớn các mô tả sản phẩm trực tuyến, thay vì phải so sánh mọi mô tả với nhau, LSH sẽ nhóm các mô tả tương tự vào cùng các bucket. 
# Sau đó, việc so sánh chi tiết chỉ cần thực hiện trong cùng một bucket, giúp tăng hiệu quả tìm kiếm các bản sao gần đúng[cite: 391, 392, 393, 394].

# ### **Lưu ý về Chi phí Tính toán**

# Sách cũng nhấn mạnh rằng việc loại bỏ trùng lặp có thể rất tốn kém về mặt tính toán[cite: 395]. 
# Để giải quyết vấn đề này ở quy mô lớn, các kỹ thuật như **minhashing** (để ước tính sự tương đồng một cách hiệu quả bằng các biểu diễn nhỏ hơn) và **xử lý song song** 
# (phân chia công việc trên nhiều bộ xử lý) có thể được sử dụng[cite: 395].

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def deduplicate_corpus(corpus, similarity_threshold=0.9):
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Compute pairwise similarities
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    # Find duplicates
    duplicates = set()
    for i in range(len(corpus)):
        for j in range(i + 1, len(corpus)):
            if similarity_matrix[i, j] > similarity_threshold:
                duplicates.add(j)
    
    # Create deduplicated corpus
    deduplicated_corpus = [doc for i, doc in enumerate(corpus) if i not in duplicates]
    
    return deduplicated_corpus

# Example usage
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above the sleepy canine.",
    "The quick brown fox jumps over the lazy dog.",
    "An entirely different sentence about cats.",
]

deduplicated = deduplicate_corpus(corpus)
print(f"Original corpus size: {len(corpus)}")
print(f"Deduplicated corpus size: {len(deduplicated)}")
print("Deduplicated corpus:")
for doc in deduplicated:
    print(f"- {doc}")