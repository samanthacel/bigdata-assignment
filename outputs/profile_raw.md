# Profil Data Mentah

Baris: 231,685 | Kolom: 10 | Duplikat persis: 0

| kolom       | jenis_terdeteksi   |   non_null |   null |   null_pct |   unik | contoh            | top5                                                                                                                                       |
|:------------|:-------------------|-----------:|-------:|-----------:|-------:|:------------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| CaseNumber  | numerik (ID/angka) |     231685 |      0 |       0    | 231685 | 100101            | 100101 (1); 256419 (1); 256393 (1); 256394 (1); 256395 (1)                                                                                 |
| SLAStatus   | kategorikal        |     231685 |      0 |       0    |      2 | WithinSLA         | WithinSLA (227726); Breach (3959)                                                                                                          |
| ClosedDate  | tanggal-waktu      |     230680 |   1005 |       0.43 | 127976 | 06/03/2023 8:33   | 06/03/2023 11:27 (600); 06/03/2023 11:26 (402); 06/03/2023 11:29 (401); 06/03/2023 8:39 (401); 06/03/2023 8:50 (400)                       |
| CreatedDate | tanggal-waktu      |     231685 |      0 |       0    | 130598 | 05/03/2023 19:27  | 27/12/2023 13:02 (175); 02/08/2023 11:45 (142); 05/03/2023 21:31 (142); 14/08/2023 15:12 (140); 05/03/2023 21:11 (136)                     |
| Channel     | kategorikal        |     231685 |      0 |       0    |      2 | Case              | Case (144317); Email (87368)                                                                                                               |
| Category    | kategorikal        |     231685 |      0 |       0    |     10 | General Inquiries | Sales (23317); General Inquiries (23288); Complaints (23261); Document Requests (23195); Lease/Rent Inquiries (23182)                      |
| Status      | kategorikal        |     231685 |      0 |       0    |      5 | Closed            | Closed (230680); In Progress (720); Open (157); Waiting Other Department (121); Waiting for Client (7)                                     |
| SubCategory | kategorikal        |     231685 |      0 |       0    |     50 | Property Rules    | Post-Sale Service (4751); Rent Payment (4732); Electrical (4732); Contact Information (4723); Sales Contract (4720)                        |
| Type        | kategorikal        |     231685 |      0 |       0    |      4 | Inquiries         | Inquiries (92837); Requests (92654); Complaints (23261); Feedback (22933)                                                                  |
| AssignedTo  | kategorikal        |     231685 |      0 |       0    |     24 | Call Center       | Call Center (171402); Lease Renewals (22124); Property Valuation (16069); Real Estate Support (4342); Real Estate Financial Reports (3638) |

## Petunjuk format tanggal

{'ClosedDate': {'bagian1_gt12': 0, 'bagian2_gt12': 0, 'contoh': ['06/03/2023 8:33', '06/03/2023 8:33', '06/03/2023 8:33']}, 'CreatedDate': {'bagian1_gt12': 0, 'bagian2_gt12': 0, 'contoh': ['05/03/2023 19:27', '05/03/2023 19:27', '05/03/2023 19:27']}}
