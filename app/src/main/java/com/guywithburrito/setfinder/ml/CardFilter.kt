package com.guywithburrito.setfinder.ml

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp

/**
 * Stage 2: Card Filtering.
 * Decides if a chip represents a valid Set card.
 */
interface CardFilter {
    /**
     * Returns true if the provided 144x224 chip is likely a Set card.
     */
    fun isCard(chip: Bitmap): Boolean
    
    fun close()

    companion object {
        /**
         * Factory method to get the default implementation.
         * Implementation details like threshold and model path are hidden here.
         */
        fun getInstance(context: Context): CardFilter {
            return TFLiteCardFilter(TFLiteCardFilterModel(context, "card_filter.tflite"), threshold = 0.1f)
        }
    }
}

/**
 * Standard implementation using the CardFilterModel.
 */
class TFLiteCardFilter(
    private val model: CardFilterModel,
    private val threshold: Float
) : CardFilter {
    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))
        .build()

    override fun isCard(chip: Bitmap): Boolean {
        var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
        tensorImage.load(chip)
        tensorImage = imageProcessor.process(tensorImage)
        
        return model.getConfidence(tensorImage.buffer) >= threshold
    }

    override fun close() {
        model.close()
    }
}
