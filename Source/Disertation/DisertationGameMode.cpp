// Copyright 1998-2019 Epic Games, Inc. All Rights Reserved.

#include "DisertationGameMode.h"
#include "DisertationCharacter.h"
#include "UObject/ConstructorHelpers.h"

ADisertationGameMode::ADisertationGameMode()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnBPClass(TEXT("/Game/ThirdPersonCPP/Blueprints/ThirdPersonCharacter"));
	static ConstructorHelpers::FClassFinder<APlayerController> PlayerControllerBPClass(TEXT("/Game/ThirdPersonCPP/Blueprints/PlayerController_BP"));
	if (PlayerPawnBPClass.Class != nullptr)
	{
		DefaultPawnClass = PlayerPawnBPClass.Class;
	}

	if (PlayerControllerBPClass.Class != nullptr) {
		PlayerControllerClass = PlayerControllerBPClass.Class;
	}
}
