from docx import Document
from docx.shared import Inches
from functions import *

def create_document():

    document = Document()
    document.add_heading("Prosjeci čišćenja", 0)
    document.save("test.docx") 


create_document()