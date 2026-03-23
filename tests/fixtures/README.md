Test fixtures for OCR-related unit tests.

Derived NOAA image:
- Source PDF title: "Eating Lake Michigan Fish"
- Source: NOAA Institutional Repository
- Landing page: https://repository.library.noaa.gov/view/noaa/40220
- Direct PDF: https://repository.library.noaa.gov/view/noaa/40220/noaa_40220_DS1.pdf
- Rights: Public Domain
- `noaa_eating_lake_michigan_fish_page1.png` was rendered from page 1 of the source PDF with:
  `pdftoppm -f 1 -singlefile -png -r 144 noaa_eating_lake_michigan_fish.pdf noaa_eating_lake_michigan_fish_page1`

Additional math textbook page:
- File: `calculus_made_easy_page272.png`
- Source page: https://commons.wikimedia.org/wiki/File:Calculus_Made_Easy.pdf
- Source page image: https://en.wikisource.org/wiki/Page:Calculus_Made_Easy.pdf/272
- Direct PDF: https://upload.wikimedia.org/wikipedia/commons/3/39/Calculus_Made_Easy.pdf
- Rights: Public Domain
- Notes: `calculus_made_easy_page272.png` was rendered from page 272 of the source PDF with:
  `pdftoppm -f 272 -l 272 -singlefile -png -r 144 Calculus_Made_Easy.pdf calculus_made_easy_page272`

OpenStax textbook excerpt:
- File: `openstax_university_physics_selected_pages.pdf`
- Source title: "University Physics Volume 1"
- Source page: https://openstax.org/details/books/university-physics-volume-1
- Direct PDF: https://assets.openstax.org/oscms-prodcms/media/documents/UniversityPhysicsVol1-WEB.pdf
- Attribution: OpenStax, Rice University
- Rights: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Notes: `openstax_university_physics_selected_pages.pdf` contains extracted source pages 745, 752, and 873 from the OpenStax PDF. It was created with:
  `qpdf --empty --pages University_Physics_Volume_1_-_WEB.pdf 745,752,873 -- openstax_university_physics_selected_pages.pdf`
