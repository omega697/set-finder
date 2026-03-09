package com.guywithburrito.setfinder.ml

import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.google.common.truth.Truth.assertThat
import org.junit.Test
import org.junit.runner.RunWith
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import java.lang.StringBuilder

@RunWith(AndroidJUnit4::class)
class CardExpertModelInstrumentationTest {

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))
        .build()

    @Test
    fun predict_isAccurateForAllTestChips() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context
        
        val expert = TFLiteExpertModel(appContext)
        val assets = testContext.assets.list("") ?: emptyArray()
        
        val testChips = assets.filter { it.startsWith("test_") && it.endsWith(".jpg") && !it.contains("NONE") }
        
        val errors = StringBuilder()
        var passed = 0
        
        testChips.forEach { assetName ->
            val inputStream = testContext.assets.open(assetName)
            val bitmap = BitmapFactory.decodeStream(inputStream)
            
            var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)
            
            val predictions = expert.predict(tensorImage.buffer)
            
            // v12 Mappings: Col=0, Shp=1, Cnt=2, Pat=3
            val colIdx = argmax(predictions[0]!!)
            val shpIdx = argmax(predictions[1]!!)
            val cntIdx = argmax(predictions[2]!!)
            val patIdx = argmax(predictions[3]!!)
            
            val expected = parseLabel(assetName)
            val actual = mapToLabel(colIdx, shpIdx, cntIdx, patIdx)
            
            if (actual != expected) {
                errors.append("FAILED $assetName: expected $expected, but was $actual\n")
            } else {
                passed++
            }
        }
        
        expert.close()
        
        android.util.Log.d("CardExpertTest", "Passed $passed / ${testChips.size}")
        if (errors.isNotEmpty()) {
            throw AssertionError("Expert model accuracy test failed with ${testChips.size - passed} errors:\n$errors")
        }
    }

    private fun argmax(scores: FloatArray): Int {
        var bestIdx = 0; var maxVal = -1f
        for (i in scores.indices) { if (scores[i] > maxVal) { maxVal = scores[i]; bestIdx = i } }
        return bestIdx
    }

    private fun parseLabel(filename: String): String {
        // Format: test_ONE_RED_EMPTY_DIAMOND.jpg -> "ONE RED EMPTY DIAMOND"
        return filename.removePrefix("test_").removeSuffix(".jpg").replace("_", " ")
    }

    private fun mapToLabel(colIdx: Int, shpIdx: Int, cntIdx: Int, patIdx: Int): String {
        // Using v12 canonical mapping logic
        val colors = listOf("NONE", "RED", "GREEN", "PURPLE")
        val shapes = listOf("NONE", "OVAL", "DIAMOND", "SQUIGGLE")
        val counts = listOf("NONE", "ONE", "TWO", "THREE")
        val patterns = listOf("NONE", "SOLID", "SHADED", "EMPTY")
        
        return "${counts.getOrElse(cntIdx){"?"}} ${colors.getOrElse(colIdx){"?"}} ${patterns.getOrElse(patIdx){"?"}} ${shapes.getOrElse(shpIdx){"?"}}"
    }
}
