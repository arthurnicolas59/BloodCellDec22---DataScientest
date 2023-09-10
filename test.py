globules_blanc_path =[
    'streamlit_media/neutrophil.png',
    'streamlit_media/eosinophil.png',
    'streamlit_media/basophil.png',
    'streamlit_media/lymphocyte.png',
    'streamlit_media/monocyte.png'
    'streamlit_media/granu.png',
]

globules_blanc_text = [
    'Neutrophiles : les plus abondants dans le sang, luttent contre les infections bactériennes',
    'Éosinophiles : impliqués dans la réponse immunitaire des parasites et réactions allergiques.',
    'Basophiles :  impliqués également en cas de réactions allergiques et libèrent des substances telles que l’histamine.',
    'Lymphocytes :  jouent un rôle central dans l’immunité adaptative, ils comprennent les Lymphocytes B (production d’anti corps) et les lymphocytes T (reconnaissance et destruction des cellules infectées)',
    'Monocytes : en charge de la phagocytose des débris cellulaires et des agents pathogènes',
    'Granulocytes Immatures (promyélocytes, myélocytes, métamyélocytes) : globules blancs immatures qui sont produits en grandes quantités en cas d’infections graves, type leucémie.'
]

for text, path in zip(globules_blanc_text, globules_blanc_path):
    print(text)
    print(path)