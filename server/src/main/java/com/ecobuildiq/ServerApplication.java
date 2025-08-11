package com.ecobuildiq;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

import com.ecobuildiq.service.PolicyEvaluator;

@SpringBootApplication
public class ServerApplication {
    public static void main(String[] args) {
        SpringApplication.run(ServerApplication.class, args);
    }

    @Bean
    public PolicyEvaluator policyEvaluator() {
        return new PolicyEvaluator();
    }
}
