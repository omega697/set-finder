package com.guywithburrito.setfinder.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.google.accompanist.permissions.*

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun SetFinderNavGraph() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "welcome") {
        composable("welcome") {
            WelcomeScreen(
                onStartClicked = { navController.navigate("scanner") },
                onSettingsClicked = { navController.navigate("settings") }
            )
        }
        composable("settings") {
            SettingsScreen(
                onBackClicked = { navController.popBackStack() }
            )
        }
        composable("history") {
            HistoryScreen(
                onBackClicked = { navController.popBackStack() }
            )
        }
        composable("scanner") {
            ScannerScreen(
                onSettingsClicked = { navController.navigate("settings") },
                onHistoryClicked = { navController.navigate("history") }
            )
        }
    }
}

@OptIn(ExperimentalPermissionsApi::class)
@Composable
fun ScannerScreen(
    onSettingsClicked: () -> Unit,
    onHistoryClicked: () -> Unit
) {
    val cameraPermissionState = rememberPermissionState(android.Manifest.permission.CAMERA)
    
    if (cameraPermissionState.status.isGranted) {
        SetFinderView(
            onSettingsClicked = onSettingsClicked,
            onHistoryClicked = onHistoryClicked
        )
    } else {
        Column(
            modifier = Modifier.fillMaxSize().padding(32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            val textToShow = if (cameraPermissionState.status.shouldShowRationale) {
                "The camera is important for this app. Please grant the permission."
            } else {
                "Camera permission required for this feature to be available. Please grant the permission."
            }
            Text(textToShow, textAlign = androidx.compose.ui.text.style.TextAlign.Center)
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = { cameraPermissionState.launchPermissionRequest() }) {
                Text("Grant Camera Permission")
            }
        }
    }
}
