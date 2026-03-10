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
     * Executes inference on the provided ByteBuffer (preprocessed image).
     * Returns a map of head index to raw FloatArray probabilities.
     */
    fun predict(buffer: ByteBuffer): Map<Int, FloatArray>
    
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

    override fun predict(buffer: ByteBuffer): Map<Int, FloatArray> {
        buffer.rewind()
        val outputs = mutableMapOf<Int, Any>()
        
        for (i in 0 until interpreter.outputTensorCount) {
            val shape = interpreter.getOutputTensor(i).shape()
            outputs[i] = Array(1) { FloatArray(shape.last()) }
        }

        interpreter.runForMultipleInputsOutputs(arrayOf(buffer), outputs)

        val results = mutableMapOf<Int, FloatArray>()
        for (i in 0 until interpreter.outputTensorCount) {
            results[i] = (outputs[i] as Array<FloatArray>)[0]
        }
        return results
    }

    override fun close() {
        interpreter.close()
    }
}
