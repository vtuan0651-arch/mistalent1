# MIS Dashboard & Coding Workspace

Chào mừng bạn đến với cấu trúc dự án mẫu được khởi tạo cho **MIS (Management Information Systems)**. Dự án được thiết kế theo phong cách hiện đại, trực quan, hỗ trợ đầy đủ các tính năng cơ bản của một dashboard hệ thống thông tin quản lý.

## 📂 Cấu trúc thư mục dự án

```text
MIS/
├── index.html       # Giao diện chính của Dashboard (HTML5 Semantic)
├── css/
│   └── style.css    # CSS Variables, Dark/Light Mode, Layout & Hoạt ảnh
└── js/
    └── app.js       # Xử lý Logic (Theme, Canvas Chart, CRUD Table, Toast)
```

## ✨ Tính năng nổi bật đã tích hợp

1. **Giao diện hiện đại & Premium**: Thiết kế tối giản, sạch sẽ sử dụng font chữ Google Fonts (Inter & Outfit) kết hợp với các hiệu ứng chuyển đổi mượt mà và bóng mờ cao cấp.
2. **Hỗ trợ Giao diện Sáng/Tối (Light & Dark Theme)**:
   - Tự động nhận diện thiết lập của hệ điều hành.
   - Cho phép người dùng chuyển đổi thủ công nhanh chóng bằng nút trên thanh Header và lưu lại tùy chọn vào `localStorage`.
3. **Sidebar thu gọn linh hoạt**: Có thể thu gọn để tối ưu không gian làm việc trên máy tính hoặc hiển thị dạng menu trượt trên thiết bị di động.
4. **Biểu đồ động (Interactive Line Chart)**: Vẽ trực tiếp trên thẻ `<canvas>` bằng HTML5 Canvas API với màu sắc tự động thay đổi theo chủ đề Light/Dark.
5. **Quản lý dữ liệu trực tiếp (CRUD UI)**:
   - Có thể thêm bản ghi mới bằng nút **Add New Record** hiển thị Modal nhập liệu cực đẹp.
   - Sửa đổi trực tiếp hoặc xóa dữ liệu trực tiếp trên bảng biểu kèm thông báo Toast và hiệu ứng chuyển động tự nhiên.
6. **Responsive hoàn toàn**: Tương thích tốt trên màn hình máy tính bàn, laptop, máy tính bảng và điện thoại di động.

## 🚀 Cách chạy dự án

Bạn có thể chạy dự án này bằng một trong các cách sau:
1. **Chạy trực tiếp**: Double-click vào file `index.html` để mở bằng bất kỳ trình duyệt nào.
2. **VS Code Live Server (Khuyên dùng)**: Mở thư mục này trong VS Code, nhấn chuột phải vào `index.html` và chọn **Open with Live Server** để xem cập nhật thời gian thực khi viết code.
3. **Node.js Local Server**: Chạy lệnh sau trong Terminal tại thư mục này để khởi chạy một local server nhanh:
   ```bash
   npx browser-sync start --server --files "index.html, css/*.css, js/*.js"
   ```

*Bây giờ bạn có thể bắt đầu mở các file này trong trình soạn thảo code của mình để tiếp tục phát triển dự án MIS của bạn!*
