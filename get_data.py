from mp_api.client import MPRester
api_key = 'GPPXFjpPCxMe6U8Uz9CQa4TEhNbaaPCY'


def write_to_text_file(string):
    with open('data/data.txt', 'a') as f:
        f.write(string)
    


def get_ternary_semiconductors():
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(num_elements=3, band_gap = (0.3,3),fields=['band_gap','structure']  )
        for i in range(len(docs)):
            write_to_text_file(str(docs[i].structure.reduced_formula) + ","+ str(docs[i].band_gap) + "\n")


get_ternary_semiconductors()
