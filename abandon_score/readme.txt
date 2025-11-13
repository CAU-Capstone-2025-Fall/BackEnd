.venv) PS D:\workspace\Capstone\backend\abandon_score> python make_label.py
✅ A1(사육경험) 컬럼: A1
✅ B3(유기충동경험) 컬럼: B3

[INFO] 전체 응답자 1014명 중 반려동물 사육 경험자 866명
  ├─ 그중 '유기 충동 설문에 응답한 사람' 507명
  └─ '유기 충동 응답이 비어있거나 비정상' 359명 (제외됨)

📊 '유기 충동 경험' 분포 (0=없음, 1=가끔, 2=자주):
impulse
0    291
1    206
2     10
Name: count, dtype: int64

🎯 최종 Y 생성 완료 — 저장됨: Y_impulse_filtered.csv (n=507)
🚨 A, B 데이터도 df_valid.index 기준으로 필터링해야 순서가 일치합니다.

test 용 데이터는 키워본 경험이 있으나, 유기 충동 경험을 공란으로 작성한 사람 데이터임.

####################################################################
####################################################################


(.venv) PS D:\workspace\Capstone\backend\abandon_score> python make_unlabel.py
✅ A1(사육경험): A1
✅ B3(유기충동): B3

[INFO] 전체 응답자: 1014
       ├─ 반려동물 사육 경험자: 866
       └─ 그 중 '유기 충동 미응답자' (테스트셋): 359

📁 저장 완료 → data/survey_unlabeled_testset.xlsx (행 359)
🧩 주의: 이 데이터는 label이 없으므로 A,B feature 생성 시 참조용으로만 사용하세요.
(.venv) PS D:\workspace\Capstone\backend\abandon_score> 


####################################################################
####################################################################

✅ 실제 컬럼명 샘플: ['ID', 'SQ1', 'SQ2', 'SQ2_R', 'SQ3_2', 'SQ3_2_R', 'A1', 'A1_1_1', 'A1_1_2', 'A1_1_3']
[INFO] 전체 1014명 중 반려동물 사육 경험자 866명 유지

📦 분리 결과:
 - A_labeled: 507명 (B3 유기충동 응답 있음)
 - A_unlabeled: 359명 (B3 공란)

🎯 완료 — 저장됨:
 - data/A_labeled.csv
 - data/A_unlabeled.csv


 