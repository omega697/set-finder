package com.guywithburrito.setfinder

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.guywithburrito.setfinder.card.SetCard
import com.guywithburrito.setfinder.ml.CardIdentifier
import com.guywithburrito.setfinder.ml.CardModelMapper
import com.guywithburrito.setfinder.ml.TFLiteCardExpert
import com.guywithburrito.setfinder.ml.TFLiteCardFilter
import com.guywithburrito.setfinder.ml.TFLiteCardFilterModel
import com.guywithburrito.setfinder.ml.TFLiteCardIdentifier
import com.guywithburrito.setfinder.ml.TFLiteExpertModel
import com.google.common.truth.Truth.assertThat
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.lang.StringBuilder

@RunWith(AndroidJUnit4::class)
class ModelAccuracyTest {

    private lateinit var identifier: CardIdentifier

    @Before
    fun setUp() {
        OpenCVLoader.initDebug()
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        
        val filterModel = TFLiteCardFilterModel(context)
        val expertModel = TFLiteExpertModel(context)
        
        // Use implementations for testing configuration - relaxed filter for full set accuracy check
        identifier = TFLiteCardIdentifier(
            TFLiteCardFilter(filterModel, threshold = 0.0001f),
            TFLiteCardExpert(expertModel, CardModelMapper.V12)
        )
    }

    @Test
    fun testIdentificationAccuracyAcrossAllAssets() {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        
        val logBuilder = StringBuilder()
        var cardsTested = 0; var cardsPassed = 0
        var nonCardsTested = 0; var nonCardsPassed = 0
        
        // 1. Test Cards
        testContext.assets.list("chips/cards")?.forEach { assetName ->
            if (!assetName.endsWith(".jpg") || assetName.contains("NONE")) return@forEach
            cardsTested++
            
            val bitmap = loadAssetAsRGB("chips/cards/$assetName")
            val result = identifier.identifyCard(bitmap)
            val expected = SetCardTestUtils.parseLabelFromFilename(assetName)
            
            val pass = result == expected
            if (pass) cardsPassed++
            logBuilder.append(String.format("cards/%s: exp=%s, act=%s -> %s\n", assetName, expected, result, if (pass) "PASS" else "FAIL"))
        }

        // 2. Test Non-Cards
        testContext.assets.list("chips/non_cards")?.forEach { assetName ->
            if (!assetName.endsWith(".jpg")) return@forEach
            nonCardsTested++
            
            val bitmap = loadAssetAsRGB("chips/non_cards/$assetName")
            val result = identifier.identifyCard(bitmap)
            
            val pass = result == null
            if (pass) nonCardsPassed++
            logBuilder.append(String.format("non_cards/%s: exp=null, act=%s -> %s\n", assetName, result, if (pass) "PASS" else "FAIL"))
        }
        
        val totalPassed = cardsPassed + nonCardsPassed
        val totalTested = cardsTested + nonCardsTested
        
        android.util.Log.d("AccuracyTest", "Results:\n$logBuilder")
        android.util.Log.d("AccuracyTest", "Passed $totalPassed / $totalTested")
        
        identifier.close()
        assertThat(totalPassed).isEqualTo(totalTested)
    }

    private fun loadAssetAsRGB(path: String): Bitmap {
        val testContext = InstrumentationRegistry.getInstrumentation().context
        val inputStream = testContext.assets.open(path)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        
        val matRGBA = Mat()
        Utils.bitmapToMat(bitmap, matRGBA)
        val matRGB = Mat()
        Imgproc.cvtColor(matRGBA, matRGB, Imgproc.COLOR_RGBA2RGB)
        
        val finalBmp = Bitmap.createBitmap(matRGB.cols(), matRGB.rows(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(matRGB, finalBmp)
        
        matRGBA.release(); matRGB.release()
        return finalBmp
    }
}
