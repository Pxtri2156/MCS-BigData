# Đồ án cuối kì môn BigData
Đồ án hướng tới việc áp dụng các kĩ PySpark trong các bài toán học máy. 
## Problem
Tích hợp dữ liệu tế bào đơn đa phương thức. Bài toán được thực hiện dựa trên cuộc thi [NeurIPS2022 Open Problems - Multimodal Single-Cell Integration](https://www.kaggle.com/competitions/open-problems-multimodal)
Bài toán yêu cầu dự đoán thông Protein từ thông tin RNA trong mỗi tế bào, bài toán có thể được hiểu như là một bài toán dự đoán bảng từ một bảng dữ liệu khác.
## Dữ liệu
Dữ liệu được trích ra trong tập CITEseq từ dữ liệu gốc [tại đây](https://www.kaggle.com/competitions/open-problems-multimodal/data). Tuy nhiên tại vì PySpark không đọc được dữ liệu trong định dạng `h5`. Nên tôi đã chuyển sang định dạng `csv` bằng tệp `util/convert_h5_csv.py`

## Rút trích dữ liệu
Tập dữ liệu CITEseq có số chiều rất lớn là 22,000 chiều, vì vậy [TSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) được áp dung
để giảm số chiều của vector dữ liệu đầu vào còn 200 chiều. 

## Mô hình
Trong đồ án này, tôi sử dụng 3 loại mô hình khác nhau của PySpark là: Linear regression, Gradient-boosted tree regression và Factorization machines. Mỗi loại mô hình, cần phải huấn luyện 140 mô hình tương ứng với 140 cột của vector đầu ra

## Source code
* Tệp `train.py` dùng để huấn luyện 3 mô hình trên. 
* Tệp `EDA.ipynb` dùng để phân tích và khám phá dữ liệu.

