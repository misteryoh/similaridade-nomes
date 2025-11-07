import pandas as pd
import numpy as np
import re
from unidecode import unidecode
from collections import Counter
import string


class SimilaridadeCNPJ:
    def __init__(self):
        self.stopwords_pt = {
            "ltda",
            "me",
            "epp",
            "sa",
            "s/a",
            "comercio",
            "de",
            "do",
            "da",
            "dos",
            "das",
            "em",
            "para",
            "com",
            "&",
            "e",
            "-",
            "empresa",
            "industrial",
            "servicos",
            "comercial",
            "distribuidora",
            "distribuicao",
            "importacao",
            "exportacao",
            "representacoes",
            "representacao",
            "sociedade",
            "limitada",
        }

    def preprocessar_texto(self, texto):
        """Faz a limpeza e normalização do texto"""
        if pd.isna(texto):
            return ""

        # Converter para minúsculas e remover acentos
        texto = unidecode(str(texto).lower())

        # Remover pontuação e caracteres especiais
        texto = re.sub(r"[^\w\s]", " ", texto)

        # Remover números isolados (mas manter números que fazem parte de palavras)
        texto = re.sub(r"\b\d+\b", " ", texto)

        # Remover espaços extras
        texto = re.sub(r"\s+", " ", texto).strip()

        return texto

    def preprocessamento_avancado(self, texto):
        # Remover sufixos comuns
        sufixos = [" ltda", " me", " eireli", " s/a", " sa"]
        for sufixo in sufixos:
            texto = texto.replace(sufixo, "")

        # Normalizar abreviações
        abreviacoes = {
            "ind ": "industria ",
            "com ": "comercio ",
            "cia ": "companhia ",
            "repr ": "representacao ",
        }

        for abrev, completa in abreviacoes.items():
            texto = texto.replace(abrev, completa)

        return texto

    def tokenizar(self, texto):
        """Divide o texto em tokens e remove stopwords"""
        tokens = texto.split()

        # Filtrar stopwords e tokens muito curtos
        tokens_filtrados = [
            token
            for token in tokens
            if token not in self.stopwords_pt and len(token) > 2
        ]

        return tokens_filtrados

    def calcular_similaridade_jaccard(self, tokens1, tokens2):
        """Calcula similaridade de Jaccard entre dois conjuntos de tokens"""
        set1 = set(tokens1)
        set2 = set(tokens2)

        if not set1 and not set2:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def calcular_similaridade_cosine(self, tokens1, tokens2):
        """Calcula similaridade de Cosseno entre dois conjuntos de tokens"""
        counter1 = Counter(tokens1)
        counter2 = Counter(tokens2)

        # Criar vocabulário único
        all_tokens = set(tokens1) | set(tokens2)

        if not all_tokens:
            return 0.0

        # Calcular vetores
        vec1 = [counter1[token] for token in all_tokens]
        vec2 = [counter2[token] for token in all_tokens]

        # Calcular produto escalar
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calcular normas
        norm1 = sum(a**2 for a in vec1) ** 0.5
        norm2 = sum(b**2 for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def encontrar_melhor_match(self, nome_certificado, df_empresas, top_n=3):
        """Encontra os melhores matches para um nome de certificado"""
        nome_processed = self.preprocessar_texto(nome_certificado)
        nome_processed = self.preprocessamento_avancado(nome_processed)
        tokens_certificado = self.tokenizar(nome_processed)

        if not tokens_certificado:
            return None

        resultados = []

        for _, empresa in df_empresas.iterrows():
            nome_empresa_processed = self.preprocessar_texto(empresa["razao_social"])
            nome_empresa_processed = self.preprocessamento_avancado(
                nome_empresa_processed
            )
            tokens_empresa = self.tokenizar(nome_empresa_processed)

            if not tokens_empresa:
                continue

            # Calcular múltiplas métricas de similaridade
            jaccard_sim = self.calcular_similaridade_jaccard(
                tokens_certificado, tokens_empresa
            )
            cosine_sim = self.calcular_similaridade_cosine(
                tokens_certificado, tokens_empresa
            )

            # Score combinado (ponderado)
            score_final = 0.6 * cosine_sim + 0.4 * jaccard_sim

            resultados.append(
                {
                    "cnpj": empresa["cnpj"],
                    "razao_social_original": empresa["razao_social"],
                    "razao_social_processada": nome_empresa_processed,
                    "score_similaridade": score_final,
                    "similaridade_jaccard": jaccard_sim,
                    "similaridade_cosine": cosine_sim,
                }
            )

        # Ordenar por score e retornar os melhores
        resultados.sort(key=lambda x: x["score_similaridade"], reverse=True)
        return resultados[:top_n]

    def processar_lote(self, df_certificados, df_empresas, threshold=0.7):
        """Processa um lote de certificados e retorna os matches"""
        resultados = []

        for _, certificado in df_certificados.iterrows():
            nome_certificado = certificado["nome_empresa"]

            matches = self.encontrar_melhor_match(nome_certificado, df_empresas)

            if matches and matches[0]["score_similaridade"] >= threshold:
                melhor_match = matches[0]
                resultados.append(
                    {
                        "nome_certificado": nome_certificado,
                        "cnpj_encontrado": melhor_match["cnpj"],
                        "razao_social_match": melhor_match["razao_social_original"],
                        "score_similaridade": melhor_match["score_similaridade"],
                        "status": "MATCH",
                    }
                )
            else:
                resultados.append(
                    {
                        "nome_certificado": nome_certificado,
                        "cnpj_encontrado": None,
                        "razao_social_match": None,
                        "score_similaridade": matches[0]["score_similaridade"]
                        if matches
                        else 0,
                        "status": "SEM_MATCH",
                    }
                )

        return pd.DataFrame(resultados)


# Exemplo de uso
def exemplo_uso():
    # Dados de exemplo
    dados_empresas = {
        "cnpj": [
            "12.345.678/0001-90",
            "98.765.432/0001-10",
            "11.223.344/0001-55",
            "70.939.558/0001-12",
        ],
        "razao_social": [
            "EMPRESA XYZ LTDA",
            "COMERCIO DE PRODUTOS ABC EIRELI",
            "INDUSTRIA E COMERCIO DE MAQUINAS SA",
            "CAETE S A INDUSTRIA E COMERCIO DE BEBIDAS",
        ],
    }

    dados_certificados = {
        "nome_empresa": [
            "Empresa XYZ Limitada",
            "Comercio Produtos ABC",
            "Ind. Com. Maquinas S.A.",
            "Loja nao existente Teste",
            "CAETE S/A",
        ]
    }

    df_empresas = pd.DataFrame(dados_empresas)
    df_certificados = pd.DataFrame(dados_certificados)

    # Usar o algoritmo
    similaridade_cnpj = SimilaridadeCNPJ()

    # Processar um certificado específico
    resultado = similaridade_cnpj.encontrar_melhor_match("CAETE S/A", df_empresas)

    print("Melhores matches:")
    for match in resultado:
        print(f"CNPJ: {match['cnpj']}")
        print(f"Razão Social: {match['razao_social_original']}")
        print(f"Score: {match['score_similaridade']:.4f}")
        print("---")

    # Processar lote completo
    resultados_lote = similaridade_cnpj.processar_lote(df_certificados, df_empresas)
    print("\nResultados do lote:")
    print(resultados_lote)


if __name__ == "__main__":
    exemplo_uso()
