from aiaioo.configuration import *
from aiaioo.data_manager import *
from aiaioo.components import *
from aiaioo.builders import *

## Running training

if __name__ == "__main__":
    
    name = "toy_ae"
    
    #name = "marsanma_chat"
    #directory = 'marsanma'

    training_data_path = "naive-audio-train_300000.txt"
    model_path = "models"+os.path.sep+name+".model"
    weights = None #[1., 1., 1., 1., 1., 5.]
    
    ## First, we prepare the data
    
    input_ = Input(training_data_path)
    #input_char_lang, input_lang, output_lang, pairs = input_.prepare_data()
    #if trim > 0:
    #    input_lang.trim(trim)
    #    output_lang.trim(trim)
    #
    #if len(output_lang.index2word) < 20:
    #    print("Output dictionary", output_lang.index2word)
        
    ## Next, we build the models
    model_builder = Seq2SeqModelBuilder()
    #model_builder = LuongAttentionSeq2SeqModelBuilder()
#     model_builder = BahdanauAttentionSeq2SeqModelBuilder()
    predictor = model_builder.get_predictor(input_, weights)
    
    renderer = SequenceRenderer()
    trainer = Trainer(renderer)
    trainer.debug = False
    trainer.train_iterations(predictor, input_, model_path)
    
