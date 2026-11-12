
extends CanvasLayer

const API_CHATBOT = "http://127.0.0.1:8000/faq/search"
const API_REGRESSAO = "http://127.0.0.1:8000/predict/value"

const FUNDO = Color("#171717")
const PAINEL = Color("#242424")
const BORDA = Color("#383838")
const BOTAO = Color("#353535")
const BOTAO_HOVER = Color("#484848")
const VERDE = Color("#8DB9A0")
const TEXTO = Color("#F2F2F2")
const TEXTO_SECUNDARIO = Color("#AAAAAA")

var requisicao_chat: HTTPRequest
var requisicao_regressao: HTTPRequest

var painel_chat: PanelContainer
var historico: RichTextLabel
var campo_pergunta: LineEdit
var botao_enviar: Button
var botao_abrir_chat: Button

var resultado_previsao: Label
var botao_prever: Button

#dados simulados

var distancia_percorrida_unidades: float = 1200.0
var dano_recebido: int = 30
var itens_coletados: int = 8
var salas_exploradas: int = 6
var tempo_decorrido_min: float = 12.5


func _ready() -> void:
	layer = 10
	process_mode = Node.PROCESS_MODE_ALWAYS

	requisicao_chat = HTTPRequest.new()
	requisicao_chat.timeout = 10.0
	add_child(requisicao_chat)
	requisicao_chat.request_completed.connect(_ao_receber_resposta_chat)

	requisicao_regressao = HTTPRequest.new()
	requisicao_regressao.timeout = 10.0
	add_child(requisicao_regressao)
	requisicao_regressao.request_completed.connect(_ao_receber_previsao)

	_criar_interface()


func _estilo_painel() -> StyleBoxFlat:
	var estilo = StyleBoxFlat.new()
	estilo.bg_color = PAINEL
	estilo.border_color = BORDA
	estilo.set_border_width_all(1)
	estilo.set_corner_radius_all(8)
	estilo.content_margin_left = 10
	estilo.content_margin_right = 10
	estilo.content_margin_top = 9
	estilo.content_margin_bottom = 9
	return estilo


func _estilo_botao(cor: Color) -> StyleBoxFlat:
	var estilo = StyleBoxFlat.new()
	estilo.bg_color = cor
	estilo.set_corner_radius_all(6)
	estilo.content_margin_left = 7
	estilo.content_margin_right = 7
	estilo.content_margin_top = 5
	estilo.content_margin_bottom = 5
	return estilo


func _estilo_campo() -> StyleBoxFlat:
	var estilo = StyleBoxFlat.new()
	estilo.bg_color = FUNDO
	estilo.border_color = BORDA
	estilo.set_border_width_all(1)
	estilo.set_corner_radius_all(6)
	estilo.content_margin_left = 7
	estilo.content_margin_right = 7
	estilo.content_margin_top = 5
	estilo.content_margin_bottom = 5
	return estilo


func _estilizar_botao(botao: Button, destaque: bool = false) -> void:
	botao.add_theme_stylebox_override("normal", _estilo_botao(VERDE if destaque else BOTAO))
	botao.add_theme_stylebox_override("hover", _estilo_botao(BOTAO_HOVER))
	botao.add_theme_stylebox_override("pressed", _estilo_botao(Color("#202020")))
	botao.add_theme_color_override("font_color", Color("#151515") if destaque else TEXTO)
	botao.add_theme_color_override("font_hover_color", TEXTO)
	botao.add_theme_font_size_override("font_size", 11)
	botao.focus_mode = Control.FOCUS_NONE
	botao.custom_minimum_size = Vector2(0, 30)


func _criar_interface() -> void:
	var coluna = VBoxContainer.new()
	coluna.set_anchors_preset(Control.PRESET_TOP_RIGHT)
	coluna.position = Vector2(-250, 15)
	coluna.size = Vector2(230, 0)
	coluna.add_theme_constant_override("separation", 6)
	add_child(coluna)

	_criar_painel_regressao(coluna)
	_criar_painel_chat(coluna)


func _criar_painel_regressao(container: VBoxContainer) -> void:
	var painel = PanelContainer.new()
	painel.add_theme_stylebox_override("panel", _estilo_painel())
	container.add_child(painel)

	var coluna = VBoxContainer.new()
	coluna.add_theme_constant_override("separation", 5)
	painel.add_child(coluna)

	var titulo = Label.new()
	titulo.text = "Previsão da missão"
	titulo.add_theme_color_override("font_color", TEXTO)
	titulo.add_theme_font_size_override("font_size", 13)
	coluna.add_child(titulo)

	resultado_previsao = Label.new()
	resultado_previsao.text = "Aguardando previsão"
	resultado_previsao.add_theme_color_override("font_color", VERDE)
	resultado_previsao.add_theme_font_size_override("font_size", 12)
	resultado_previsao.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	coluna.add_child(resultado_previsao)

	botao_prever = Button.new()
	botao_prever.text = "Prever tempo"
	_estilizar_botao(botao_prever, true)
	botao_prever.pressed.connect(_testar_previsao)
	coluna.add_child(botao_prever)


func _criar_painel_chat(container: VBoxContainer) -> void:
	botao_abrir_chat = Button.new()
	botao_abrir_chat.text = "Assistente Campos Zone"
	botao_abrir_chat.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	_estilizar_botao(botao_abrir_chat)
	container.add_child(botao_abrir_chat)

	painel_chat = PanelContainer.new()
	painel_chat.add_theme_stylebox_override("panel", _estilo_painel())
	painel_chat.custom_minimum_size = Vector2(230, 0)
	painel_chat.visible = false
	container.add_child(painel_chat)

	var coluna = VBoxContainer.new()
	coluna.add_theme_constant_override("separation", 6)
	painel_chat.add_child(coluna)

	var titulo = Label.new()
	titulo.text = "Assistente Campos Zone"
	titulo.add_theme_color_override("font_color", TEXTO)
	titulo.add_theme_font_size_override("font_size", 13)
	coluna.add_child(titulo)

	historico = RichTextLabel.new()
	historico.bbcode_enabled = true
	historico.custom_minimum_size = Vector2(0, 125)
	historico.scroll_active = true
	historico.text = "Olá! Como posso ajudar?"
	historico.add_theme_color_override("default_color", TEXTO)
	historico.add_theme_font_size_override("normal_font_size", 11)
	historico.add_theme_stylebox_override("normal", _estilo_campo())
	coluna.add_child(historico)

	var linha = HBoxContainer.new()
	linha.add_theme_constant_override("separation", 5)
	coluna.add_child(linha)

	campo_pergunta = LineEdit.new()
	campo_pergunta.placeholder_text = "Digite sua pergunta..."
	campo_pergunta.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	campo_pergunta.focus_mode = Control.FOCUS_ALL
	campo_pergunta.add_theme_stylebox_override("normal", _estilo_campo())
	campo_pergunta.add_theme_color_override("font_color", TEXTO)
	campo_pergunta.add_theme_font_size_override("font_size", 11)
	campo_pergunta.text_submitted.connect(_enviar_pergunta)
	linha.add_child(campo_pergunta)

	botao_enviar = Button.new()
	botao_enviar.text = "Enviar"
	_estilizar_botao(botao_enviar)
	botao_enviar.pressed.connect(func(): _enviar_pergunta(campo_pergunta.text))
	linha.add_child(botao_enviar)

	botao_abrir_chat.pressed.connect(_alternar_chat)


func _alternar_chat() -> void:
	var abrir = not painel_chat.visible
	painel_chat.visible = abrir

	if abrir:
		get_tree().paused = true
		campo_pergunta.call_deferred("grab_focus")
	else:
		campo_pergunta.release_focus()
		get_viewport().gui_release_focus()
		get_tree().paused = false


func _enviar_pergunta(texto: String) -> void:
	texto = texto.strip_edges()

	if texto.is_empty():
		return

	if requisicao_chat.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		return

	historico.append_text("\n\nVocê: " + texto)
	campo_pergunta.clear()
	botao_enviar.disabled = true

	var dados = {"pergunta": texto}
	var erro = requisicao_chat.request(
		API_CHATBOT,
		["Content-Type: application/json"],
		HTTPClient.METHOD_POST,
		JSON.stringify(dados)
	)

	if erro != OK:
		historico.append_text("\nNão foi possível enviar a pergunta.")
		botao_enviar.disabled = false


func _ao_receber_resposta_chat(
	resultado: int,
	codigo_http: int,
	_cabecalhos: PackedStringArray,
	corpo: PackedByteArray
) -> void:
	botao_enviar.disabled = false

	if resultado != HTTPRequest.RESULT_SUCCESS:
		historico.append_text("\nErro de comunicação com a API.")
		return

	var dados = JSON.parse_string(corpo.get_string_from_utf8())

	if not dados is Dictionary:
		historico.append_text("\nResposta inválida da API.")
		return

	if codigo_http != 200:
		historico.append_text("\nErro HTTP: " + str(codigo_http))
		return

	var resposta = str(dados.get("resposta", "Não foi possível obter uma resposta."))
	historico.append_text("\n\nAssistente: " + resposta)


func _testar_previsao() -> void:
	if requisicao_regressao.get_http_client_status() != HTTPClient.STATUS_DISCONNECTED:
		return

	var dados = {
		"distancia_percorrida_unidades": distancia_percorrida_unidades,
		"dano_recebido": dano_recebido,
		"itens_coletados": itens_coletados,
		"salas_exploradas": salas_exploradas,
		"tempo_decorrido_min": tempo_decorrido_min
	}

	botao_prever.disabled = true
	resultado_previsao.text = "Calculando..."

	var erro = requisicao_regressao.request(
		API_REGRESSAO,
		["Content-Type: application/json"],
		HTTPClient.METHOD_POST,
		JSON.stringify(dados)
	)

	if erro != OK:
		resultado_previsao.text = "Falha ao enviar dados."
		botao_prever.disabled = false


func _ao_receber_previsao(
	resultado: int,
	codigo_http: int,
	_cabecalhos: PackedStringArray,
	corpo: PackedByteArray
) -> void:
	botao_prever.disabled = false

	if resultado != HTTPRequest.RESULT_SUCCESS:
		resultado_previsao.text = "Erro de comunicação."
		return

	var resposta = JSON.parse_string(corpo.get_string_from_utf8())

	if not resposta is Dictionary:
		resultado_previsao.text = "Resposta inválida."
		return

	if codigo_http != 200:
		resultado_previsao.text = "Erro HTTP: " + str(codigo_http)
		return

	var previsao = resposta.get("previsao", {})
	var minutos = float(previsao.get("tempo_restante_minutos", 0.0))

	resultado_previsao.text = "Conclusão estimada: %.1f min" % minutos
