// Fill out your copyright notice in the Description page of Project Settings.


#include "Location.h"

// Sets default values
ALocation::ALocation()
{

}

// Called when the game starts or when spawned
void ALocation::BeginPlay()
{
	Super::BeginPlay();
	
}

// Called every frame
void ALocation::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);
}

void ALocation::SetTitle(FString NewTitle)
{
	Title = NewTitle;
}

void ALocation::SetDescription(FString NewDescription)
{
	Description = NewDescription;
}

