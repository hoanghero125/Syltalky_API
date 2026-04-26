# Speaker Assets

Thư mục này chứa các giọng nói mẫu (preset speakers) cho hệ thống TTS.

## Cấu trúc thư mục:
- `assets/speakers/metadata.json`: Chứa thông tin mô tả về các speaker.
- `assets/speakers/<speaker_id>/ref.wav`: File âm thanh mẫu của speaker đó.

## Cách thêm Speaker mới:
1. Tạo một thư mục mới: `mkdir assets/speakers/new_speaker`
2. Copy file âm thanh mẫu (WAV, 24kHz hoặc cao hơn) vào: `assets/speakers/new_speaker/ref.wav`
3. Cập nhật thông tin vào `metadata.json`.

## Cách sử dụng:
Sử dụng script `use_speakers.py` ở thư mục gốc để kiểm tra các giọng nói này.
Hoặc thông qua API `/v1/presets` để lấy danh sách.
