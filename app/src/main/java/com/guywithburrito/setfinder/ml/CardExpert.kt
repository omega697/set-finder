package com.guywithburrito.setfinder.ml

import android.content.Context
import android.graphics.Bitmap
import com.guywithburrito.setfinder.card.SetCard
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp

/**
 * Stage 3: Card Identification.
 * Identifies the card attributes from a verified card chip.
 */
interface CardExpert {
    /**
     * Identifies the attributes of a 144x224 card chip.
     */
    fun identify(chip: Bitmap): SetCard?
    
    fun close()

    companion object {
        /**
         * Factory method to get the default implementation.
         * Implementation details like model path and mapper version are hidden here.
         */
        fun getInstance(context: Context): CardExpert {
            return TFLiteCardExpert(TFLiteExpertModel(context, "set_card_model_final.tflite"), CardModelMapper.V12)
        }
    }
}

/**
 * Standard implementation using the CardExpertModel and CardModelMapper.
 */
class TFLiteCardExpert(
    private val model: CardExpertModel,
    private val mapper: CardModelMapper
) : CardExpert {
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))
        .build()

    override fun identify(chip: Bitmap): SetCard? {
        var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(chip)
        tensorImage = imageProcessor.process(tensorImage)
        
        val predictions = model.predict(tensorImage.buffer)
        
        // Mappings based on model version (defaults to v12 logic)
        val colIdx = mapper.argmax(predictions[0]!!)
        val cntIdx = mapper.argmax(predictions[1]!!)
        val patIdx = mapper.argmax(predictions[2]!!)
        val shpIdx = mapper.argmax(predictions[3]!!)
        
        return mapper.mapIndices(colIdx, shpIdx, cntIdx, patIdx)
    }

    override fun close() {
        model.close()
    }
}
