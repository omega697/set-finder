package com.guywithburrito.setfinder

import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy

class SetAnalyzer : ImageAnalysis.Analyzer {
    override fun analyze(image: ImageProxy) {
        // Don't actually do anything yet.
        image.close()
    }
}