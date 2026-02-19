from pptx import Presentation
import sys

# Set UTF-8 encoding for output
sys.stdout.reconfigure(encoding='utf-8')

prs = Presentation(r'k:\download\PPT_final.pptx')

for i, slide in enumerate(prs.slides):
    print(f'\n{"="*60}')
    print(f'SLIDE {i+1}')
    print(f'{"="*60}\n')

    for shape in slide.shapes:
        if hasattr(shape, 'text') and shape.text.strip():
            print(shape.text)
            print()
