from sentence_transformers import SentenceTransformer

model = SentenceTransformer("shibing624/text2vec-base-chinese")
model.save("./models/text2vec-base-chinese")