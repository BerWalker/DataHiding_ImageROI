from pathlib import Path

import argparse
import cv2

CHANNEL = 1
COLUNA = 30
LINHA = 30


# Faz um bitshift em uma string
def pad_binary(b: str) -> str:
	if len(b) == 8:
		return b
	if len(b) < 8:
		diff = 8 - len(b)
		return "0" * diff + b


# Faz o processo de esteganografia em uma imagem
def steg(text: str, img_in: Path, img_out: Path) -> None:
	img = cv2.imread(str(img_in), cv2.IMREAD_COLOR)
	linhas, colunas, canais = img.shape
	if not (linhas > LINHA and COLUNA < colunas):
		raise EOFError("Dimensão da imagem não permite")
	print(colunas, linhas, canais)
	if type(img) == None:
		raise EOFError(f"Arquivo '{img_in}' não encontrado")

	total_bits = 0
	ch_to_hide = ""
	for ch in text:
		# Removendo "0b" no início da conversão
		# Ex: 0b100101 -> 100101
		bin_char = bin(ord(ch))[2:]
		# Adiciona mais zeros se a quantidade de bits for
		# menor que 8
		bin_char = pad_binary(bin_char)
		# Concatena sequência de bits
		ch_to_hide += bin_char
		# Computa total de bits
		total_bits += len(bin_char)

	img_steg = img.copy()
	linha = LINHA
	coluna = COLUNA
	for i in range(len(ch_to_hide)):
		if coluna == colunas:
			coluna = COLUNA
			linha += 1
		# Faz a operação bitwise and com a flag 1
		if int(ch_to_hide[i]) & 1:
			# Concatena bit
			img_steg[linha, coluna, CHANNEL] = img_steg[linha, coluna, CHANNEL] | 1
		else:
			# Concatena bit
			img_steg[linha, coluna, CHANNEL] = (
				img_steg[linha, coluna, CHANNEL] & 0xFE
			)
		coluna += 1

	print(f"Escondendo mensagem '{text}' na imagem {img_out.resolve()}")
	print(f"Transformando em '{ch_to_hide}'")
	# Salva imagem no disco
	cv2.imwrite(str(img_out), img_steg)


# Faz o processo reverso da esteganografia de uma imagem
def unsteg(img_path: str, n_bytes: int) -> None:
	# Carrega imagem
	img_steg = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
	linhas, colunas, canais = img_steg.shape
	if type(img_steg) == None:
		raise EOFError(f"Arquivo '{img_path}' não encontrado")
	# Passa por todos pela quantidade de caracteres * bits (8)
	bits = ""
	linha = LINHA
	coluna = COLUNA
	for i in range(n_bytes * 8):
		if coluna == colunas:
			linha += 1
			coluna = COLUNA
		# Adiciona o bit alterado fazendo a operação bitwise and (&)
		bits += str(img_steg[linha, coluna, CHANNEL] & 1)
		coluna += 1

	last_i = 0
	msg_escondida = ""
	# Itera de 8 em 8 bits (1 byte)
	for i in range(8, len(bits) + 8, 8):
		# Pega o caractere na região last_i até i e
		# transforma para número inteiro na base 2
		char_ascii = int(bits[last_i:i], base=2)
		# Converte o valor inteiro para o caractere correspondente da
		# tabela ASCII
		char_ascii = chr(char_ascii)
		# Altera o último índice
		if last_i != i:
			last_i = i
		# Adiciona o caractere a mensagem escondida
		msg_escondida += char_ascii
	print(msg_escondida)

PARSER = argparse.ArgumentParser(prog='Least Significant Bit')
PARSER.add_argument('-i', '--input-img', required=True, help='Caminho absoluto da imagem que você quer esconder alguma informação')
PARSER.add_argument('-o', '--output-img', help='Caminho absoluto de onde você quer salvar a imagem com uma informação escondida')
PARSER.add_argument('-m', '--message', help='Texto à ser escondido na imagem')
PARSER.add_argument('-n', '--n-bytes', help='Especifica a quantidade de bytes que você quer decodificar caso a flag \'hide\' esteja setada')
PARSER.add_argument('--hide', help='Flag para indicar se você quer esconder ou revelar uma mensagem de uma imagem', action=argparse.BooleanOptionalAction)
PARSER.set_defaults(hide=False)

if __name__ == '__main__':
    args = PARSER.parse_args()
    if args.hide == True:
        steg(args.message, Path(args.input_img), Path(args.output_img))
    else:
        unsteg(Path(args.input_img), int(args.n_bytes))