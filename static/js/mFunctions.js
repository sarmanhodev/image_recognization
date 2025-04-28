function carregarImagem(){
    var imagem = $("#uploadImagem");
    console.log(imagem[0].files[0])

    if (imagem[0].files[0]){
        let formData = new FormData();
        formData.append('imagem',imagem[0].files[0]);

        $.ajax({
            type:'POST',
            url:'/analisar_imagem',
            data: formData,
            contentType: false,   // muito importante!
            processData: false,   // muito importante!
            success: function (response) {
                
                $("#recebeResultado").html("");
                console.log(response);

                var mensagem = response.prediction;
                var diretorio = response.image_path;

                var colunaResposta = `
                <div class="d-block">
                    <img class="img-fluid" width="150" src="${diretorio}">
                </div>
                
                <div class="d-block">
                    <span>${mensagem}</span>            
                </div>`;

                $("#recebeResultado").html(colunaResposta);
            },
            error: function (xhr, status, error) {
                console.error("Erro:", xhr.responseText);
            }
        });
    }    
}