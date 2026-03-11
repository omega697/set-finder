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
     * Identifies the attributes of a card chip.
     */
    fun identify(chip: Bitmap): SetCard?
    
    fun close()

    companion object {
        /**
         * Factory method to get the default implementation.
         */
        fun getInstance(context: Context): CardExpert {
            // Using v14 model and mapper. The robust mapping logic is inside CardModelMapper.
            return TFLiteCardExpert(TFLiteExpertModel(context, "attribute_expert_v14.tflite"), CardModelMapper.V14)
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
        
        // Inference returns a Map<Int, FloatArray> of indexed heads
        val predictions = model.predict(tensorImage.buffer)
        
        // The mapper handles the knowledge of which index maps to which trait
        return mapper.mapPredictions(predictions)
    }

    override fun close() {
        model.close()
    }
}
