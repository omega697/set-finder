package com.guywithburrito.setfinder.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer

/**
 * Interface for the binary classifier that distinguishes cards from background.
 */
interface CardFilterModel {
    /**
     * Returns the confidence (0.0 to 1.0) that the provided chip is a Set card.
     */
    fun getConfidence(buffer: ByteBuffer): Float
    
    fun close()
}

/**
 * TFLite implementation of the CardFilterModel.
 */
class TFLiteCardFilterModel(
    context: Context, 
    modelPath: String = "card_filter.tflite"
) : CardFilterModel {
    private val interpreter: Interpreter = Interpreter(FileUtil.loadMappedFile(context, modelPath))

    override fun getConfidence(buffer: ByteBuffer): Float {
        buffer.rewind()
        val output = Array(1) { FloatArray(1) }
        interpreter.run(buffer, output)
        val conf = output[0][0]
        android.util.Log.d("CardFilterModel", "Inference: confidence=$conf")
        return conf
    }

    override fun close() {
        interpreter.close()
    }
}
