from pdf2image import convert_from_path

images = convert_from_path('source_documents/1.pdf', poppler_path='/usr/bin')
for i, img in enumerate(images):
    img.save(f'page_{i+1}.png', 'PNG')
    print(f"Page {i+1} saved")
