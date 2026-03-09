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
class CardFilterModelInstrumentationTest {

    private val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
        .add(NormalizeOp(127.5f, 127.5f))
        .build()

    @Test
    fun getConfidence_isAccurateAcrossAllAssets() {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val testContext = InstrumentationRegistry.getInstrumentation().context
        
        val filter = TFLiteCardFilterModel(appContext)
        
        val logBuilder = StringBuilder()
        var passed = 0
        var tested = 0
        
        // 1. Test Confirmed Cards
        testContext.assets.list("chips/cards")?.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            tested++
            val inputStream = testContext.assets.open("chips/cards/$assetName")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            
            var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)
            val confidence = filter.getConfidence(tensorImage.buffer)
            
            val pass = confidence >= 0.1f
            if (pass) passed++
            logBuilder.append(String.format("chips/cards/%s: conf=%.6f -> %s\n", assetName, confidence, if (pass) "PASS" else "FAIL"))
        }

        // 2. Test Confirmed Non-Cards
        testContext.assets.list("chips/non_cards")?.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            tested++
            val inputStream = testContext.assets.open("chips/non_cards/$assetName")
            val bitmap = BitmapFactory.decodeStream(inputStream)
            
            var tensorImage = TensorImage(org.tensorflow.lite.DataType.FLOAT32)
            tensorImage.load(bitmap)
            tensorImage = imageProcessor.process(tensorImage)
            val confidence = filter.getConfidence(tensorImage.buffer)
            
            val pass = confidence < 0.1f
            if (pass) passed++
            logBuilder.append(String.format("chips/non_cards/%s: conf=%.6f -> %s\n", assetName, confidence, if (pass) "PASS" else "FAIL"))
        }
        
        filter.close()
        
        android.util.Log.d("CardFilterTest", "Results:\n$logBuilder")
        android.util.Log.d("CardFilterTest", "Passed $passed / $tested")
        
        assertThat(passed).isEqualTo(tested)
    }
}
